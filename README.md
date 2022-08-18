# CLIP Latent Exploration

Minimal working example illustrating the use of CLIP (Contrastive Language-Image Pre-Training) embeddings.    

The example uses (image, caption) pairs from [Google's Conceptual Captions dataset](https://ai.google.com/research/ConceptualCaptions/).
Data is available via the [Huggingface Hub](https://huggingface.co/datasets/conceptual_captions).
CLIP is available via the official implementation from OpenAI at https://github.com/openai/CLIP.

In the example, both images and captions are embedded using CLIP and then embeddings are projected to a low-dimensional 
space via [UMAP](https://umap-learn.readthedocs.io/en/latest/).
