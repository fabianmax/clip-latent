import os
import torch
import clip
import umap
import plotly.express as px
import pandas as pd
from datasets import load_dataset, load_from_disk, Features, Array3D, Sequence, Value

from src.clip_latent.data import fetch_images

DATA_DIR = './data'
NUM_THREADS = 20

if __name__ == '__main__':

    # Either load from disk or download the dataset
    if os.path.exists(f'{DATA_DIR}/conceptual_captions'):
        ds = load_from_disk(f'{DATA_DIR}/conceptual_captions')
    else:
        ds = load_dataset("conceptual_captions", split="train[0:10]")
        ds = ds.map(fetch_images, batched=True, batch_size=100, fn_kwargs={"num_threads": NUM_THREADS})
        ds.save_to_disk(f"{DATA_DIR}/conceptual_captions")

    # Load CLIP from pretrained weights
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

    def preprocess_txt_img(example, preprocess_fn):
        """
        Image (resize, center/crop and normalize) and Text preprocessing (tokenization).
        """
        example['image'] = preprocess_fn(example['image']).numpy()
        example['caption'] = clip.tokenize([example['caption']]).flatten()

        return example

    # Filter out images with missing information
    ds = ds.filter(lambda example: example['image'] is not None)
    ds = ds.filter(lambda example: example['caption'] is not None)

    # Preprocess images and captions
    ds = ds.remove_columns(['image_url'])
    ds_torch = ds.map(preprocess_txt_img, batched=False, fn_kwargs={"preprocess_fn": clip_preprocess})

    # Convert the huggingface dataset to torch format
    # For N-dim arrays, one needs to set the feature types explicitly (see below)
    # https://huggingface.co/docs/datasets/use_with_pytorch#ndimensional-arrays
    ds_torch = ds_torch.cast(Features({
        'caption': Sequence(feature=Value(dtype='int32')),
        'image': Array3D(shape=(3, 224, 224), dtype='float32')
    }))
    ds_torch = ds_torch.with_format("torch", device=device)

    # Extract the data from the dataset
    images = ds_torch['image'][:]
    captions = ds_torch['caption'][:]

    # Run the forward pass through the model and generate the embeddings
    with torch.no_grad():
        image_features = clip_model.encode_image(images)
        caption_features = clip_model.encode_text(captions)

        logits_per_image, logits_per_text = clip_model(images, captions)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    print("Label probs:", probs)

    # Compute the UMAP embedding in 3dims
    reducer = umap.UMAP(n_components=3)
    reducer.fit(torch.concat([image_features, caption_features], dim=0).cpu().numpy())

    umap_image_embedding = reducer.transform(image_features.cpu().numpy())
    umap_caption_embedding = reducer.transform(image_features.cpu().numpy())

    # Create a dataset that holds the embeddings for the images and captions
    umap_image_embedding_df = pd.DataFrame(umap_image_embedding, columns=['x', 'y', 'z'])
    umap_image_embedding_df['type'] = 'image'
    umap_image_embedding_df['caption'] = ds['caption'][:]

    umap_caption_embedding_df = pd.DataFrame(umap_caption_embedding, columns=['x', 'y', 'z'])
    umap_caption_embedding_df['type'] = 'caption'
    umap_caption_embedding_df['caption'] = ds['caption'][:]

    umap_embedding_df = pd.concat([umap_image_embedding_df, umap_caption_embedding_df])

    # 3d visualization that shows how caption and image embeddings are related
    fig = px.scatter_3d(umap_embedding_df, x='x', y='y', z='z', color='type', hover_data=['caption'])
    fig.show()
