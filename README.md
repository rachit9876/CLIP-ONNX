# `CLIP-ONNX`

### Run OpenAI CLIP Models Offline with ONNX Runtime on CPU, iGPU, and Edge Devices

Run OpenAI CLIP image text models fully offline using ONNX Runtime.
No dedicated GPU required. Optimized for CPUs, Intel Iris Xe, AMD APUs, laptops, mini PCs, and low power edge systems.

`CLIP-ONNX` lets you:

* Classify images using natural language prompts
* Search images with text queries
* Rank images by semantic similarity
* Run CLIP locally without internet
* Use OpenAI CLIP through ONNX for fast inference
* Deploy lightweight vision AI pipelines on Windows, Linux, or macOS

Supports:

* ONNX Runtime
* Hugging Face Transformers
* OpenAI CLIP ViT models
* CPU inference
* Offline AI workflows

---

# Features

* Fully offline inference
* ONNX exported CLIP encoders
* CPU optimized
* Works on integrated GPUs and low end systems
* Natural language image classification
* Semantic image search
* Zero shot image recognition
* Cross platform
* No cloud APIs
* Simple Python implementation
* Easily extensible for custom datasets

---

# Example Use Cases

* Local AI image tagging
* Offline photo organization
* Reverse image style matching
* AI powered desktop search
* Embedded vision systems
* Edge AI deployments
* Semantic image retrieval
* Dataset filtering
* Prompt based image ranking
* Lightweight CLIP experimentation

---

# How CLIP Works

CLIP maps both images and text into the same embedding space.

The closer the embeddings are, the more semantically related they are.

```text
Image → Vision Encoder → Image Embedding (512)
Text  → Text Encoder   → Text Embedding  (512)

Cosine Similarity
        ↓
Softmax Ranking
        ↓
Best Matching Prompt
```

This project exports both OpenAI CLIP encoders to ONNX format for fast local inference using ONNX Runtime.

---

# Repository Structure

| File              | Description                                              |
| ----------------- | -------------------------------------------------------- |
| `build.py`        | Downloads CLIP from Hugging Face and exports ONNX models |
| `test.py`         | Runs text to image similarity inference                  |
| `clip_image.onnx` | Vision encoder model                                     |
| `clip_text.onnx`  | Text encoder model                                       |

---

# Requirements

* Python 3.10+
* torch
* transformers
* onnx
* onnxruntime
* Pillow
* numpy

Install dependencies:

```bash
pip install torch transformers onnxruntime onnx Pillow numpy
```

---

# Setup

## 1. Clone Repository

```bash
git clone https://github.com/yourusername/CLIP-ONNX.git
cd CLIP-ONNX
```

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

# Build ONNX Models

Downloads CLIP weights from Hugging Face and exports ONNX models.

```bash
python build.py
```

Generated files:

```text
clip_image.onnx
clip_text.onnx
```

---

# Usage

Place images inside the project folder:

```text
image1.jpg
image2.png
image3.jpeg
```

Edit candidate prompts in `test.py`:

```python
candidate_prompts = [
    "a photo of a golden retriever puppy",
    "a black sports car parked outdoors",
    "a photo of a cat",
    "a photo of a person",
    "a photo of a landscape",
]
```

Run inference:

```bash
python test.py
```

---

# Example Output

```text
Image: dog.jpg

22.32%  a photo of a golden retriever puppy   <-- BEST
19.12%  a photo of a car
19.69%  a photo of a cat
19.71%  a photo of a person
19.16%  a photo of a landscape
```

Results are also written to:

```text
results.txt
```

---

# Best Prompting Practices

CLIP performs best with caption style prompts.

## Weak Prompt

```text
dog
```

## Better Prompt

```text
a photo of a brown dog sitting outdoors
```

More descriptive prompts usually produce stronger semantic separation.

CLIP also evaluates the entire scene, not just the main object.

---

# Supported Models

Default model:

```python
MODEL = "openai/clip-vit-base-patch32"
```

Other compatible models:

```text
openai/clip-vit-base-patch16
openai/clip-vit-large-patch14
```

Larger models improve accuracy but increase ONNX size and inference time.

---

# Run on CPU or iGPU

This repository is designed for:

* Intel Iris Xe
* AMD Radeon integrated graphics
* Low VRAM systems
* CPU only environments
* Thin laptops
* Edge hardware

No CUDA dependency required.

---

# Example Integration

```python
from test import encode_text, encode_image
import numpy as np

text_emb = encode_text(["a photo of a dog"])
image_emb = encode_image("myimage.jpg")

logits = (image_emb @ text_emb.T)[0] * 100.0

print(logits)
```

---

# Performance Notes

* Images are resized to `224×224`
* Softmax scores are relative, not absolute confidence
* Large folders should use batched inference
* FP32 models preserve embedding quality better than INT8

---

# Why ONNX?

ONNX provides:

* Faster CPU inference
* Cross platform deployment
* Hardware agnostic execution
* Easier production deployment
* Lightweight runtime requirements

Perfect for local AI applications and offline machine learning systems.

---

# SEO Keywords

CLIP ONNX, OpenAI CLIP ONNX, ONNX Runtime CLIP, offline CLIP model, local AI image classifier, CPU image recognition, zero shot image classification, semantic image search, Hugging Face CLIP ONNX, OpenAI vision model offline, CLIP ViT B32 ONNX, AI image similarity search, ONNX image embeddings, lightweight vision AI, edge AI image search, integrated GPU AI inference, CPU based computer vision, offline multimodal AI

---

# Credits

* [OpenAI CLIP](https://github.com/openai/CLIP?utm_source=chatgpt.com)
* [Hugging Face Transformers](https://huggingface.co/openai/clip-vit-base-patch32?utm_source=chatgpt.com)
* [ONNX Runtime](https://onnxruntime.ai/?utm_source=chatgpt.com)
