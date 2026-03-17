# Multimodal Embedding Alignment with LLaVA-1.6 + LoRA

A proof-of-concept project inspired by VLM2Vec.

Authors: Francisco Nicolas Noya, Pablo Gomez

## Technologies

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FF9D00?logo=huggingface&logoColor=white)](https://huggingface.co/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![Transformers](https://img.shields.io/badge/Transformers-FFD21E?logo=huggingface&logoColor=black)](https://github.com/huggingface/transformers)
[![PEFT](https://img.shields.io/badge/PEFT-2E2E2E?logo=python&logoColor=white)](https://github.com/huggingface/peft)
[![Torchvision](https://img.shields.io/badge/Torchvision-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/vision/stable/index.html)
[![BitsAndBytes](https://img.shields.io/badge/BitsAndBytes-5B5B5B?logo=nvidia&logoColor=76B900)](https://github.com/TimDettmers/bitsandbytes)

## Overview

This project fine-tunes a vision-language model so text embeddings such as "The number 8" are close to the embedding of the MNIST image of digit `8`, and far from other digits.

The notebook uses `llava-hf/llava-v1.6-mistral-7b-hf` as a dual encoder:
- Text-only forward pass -> text embedding
- Image-only forward pass -> image embedding

Alignment is learned with a diagonal InfoNCE contrastive loss.

## Project Files

- `vlm2vec_mnist_alignment.ipynb`: main notebook with the complete pipeline

## What the Notebook Does

1. Installs and imports dependencies (`transformers`, `peft`, `torchvision`, `bitsandbytes`, etc.).
2. Loads MNIST and samples one representative image per digit (0-9).
3. Loads LLaVA-1.6 with 4-bit NF4 quantization for memory efficiency.
4. Adds LoRA adapters and trains only adapter parameters.
5. Formats text-only and image-only prompts with the LLaVA chat template.
6. Extracts last-token hidden-state embeddings and L2-normalizes them.
7. Trains with diagonal InfoNCE.
8. Evaluates with a 10x10 text-image similarity matrix (Recall@1 / Precision@1).
9. Runs a zero-shot arithmetic generalization test (for example, "1 + 2" -> digit image `3`).

## Method Summary

Given normalized text embeddings $Q \in \mathbb{R}^{N \times d}$ and image embeddings $K \in \mathbb{R}^{N \times d}$:

$$S = QK^\top / \tau$$

Positives are on the diagonal. The loss is the symmetric CLIP-style InfoNCE:

$$\mathcal{L} = \frac{1}{2}\left(\text{CE}(S, I_N) + \text{CE}(S^\top, I_N)\right)$$

LoRA update form:

$$W = W_0 + \frac{\alpha}{r} BA$$

## Requirements

- Python 3.10+
- CUDA-capable GPU recommended (16GB)
- Enough disk to cache model checkpoints and MNIST

Notebook installs these packages when run:
- `transformers>=4.40.0`
- `peft>=0.10.0`
- `torchvision`
- `Pillow`
- `accelerate`
- `bitsandbytes>=0.43.0`

## How to Run

1. Open `vlm2vec_mnist_alignment.ipynb`.
2. Select a Python environment with GPU-enabled PyTorch.
3. Run cells top-to-bottom.
4. Monitor GPU memory (`nvidia-smi` cells are included).
5. Compare baseline and post-training similarity heatmaps.
6. Inspect arithmetic-query retrieval results for semantic generalization.

## Notes

- The notebook is configured for a short demonstration run (`NUM_STEPS = 5`) but can be increased.
- Image augmentation is applied during training to reduce overfitting.
- Horizontal flip is intentionally avoided because it can alter digit identity (for example `6` vs `9`).
