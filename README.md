# CLIP-ONNX

[Download Vision Model](https://github.com/rachit9876/CLIP-ONNX/releases/download/v1.0.0/clip_image.onnx) • [Download Text Model](https://github.com/rachit9876/CLIP-ONNX/releases/download/v1.0.0/clip_text.onnx)

Run OpenAI CLIP locally with ONNX Runtime on CPUs, integrated GPUs, and low power systems.

Lightweight, offline, cross platform, and optimized for fast semantic image inference without CUDA.

---

## Features

* Offline CLIP inference
* ONNX Runtime backend
* CPU and iGPU optimized
* Zero shot image classification
* Semantic image search
* Cross platform support
* Lightweight local AI pipeline

---

# Quick Start

## Clone Repository

```bash
git clone https://github.com/rachit9876/CLIP-ONNX.git
cd CLIP-ONNX
```

---

## Create Virtual Environment

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

## Install Dependencies

```bash
pip install torch transformers onnxruntime onnx Pillow numpy
```

---

## Build ONNX Models

```bash
python build.py
```

This exports:

```text
clip_image.onnx
clip_text.onnx
```

---

## Run Inference

```bash
python test.py
```

---

# Example

## Candidate Prompts

```python
candidate_prompts = [
    "a photo of a dog",
    "a sports car",
    "a cat",
    "a person",
]
```

## Output

```text
Image: dog.jpg

82.31%  a photo of a dog
7.14%   a cat
5.02%   a person
2.91%   a sports car
```

---

# Repository Structure

```text
CLIP-ONNX/
│
├── build.py          # Export CLIP models to ONNX
├── test.py           # Run similarity inference
├── clip_image.onnx   # Vision encoder
└── clip_text.onnx    # Text encoder
```

---

# Supported Models

## Default

```python
MODEL = "openai/clip-vit-base-patch32"
```

## Compatible Models

```text
openai/clip-vit-base-patch16
openai/clip-vit-large-patch14
```

---

# Supported Hardware

Designed for:

* Intel Iris Xe
* AMD integrated graphics
* CPU only systems
* Thin laptops
* Edge devices
* Mini PCs

CUDA is not required.

---

# Performance Notes

* Images are resized to `224×224`
* FP32 preserves embedding quality
* Batched inference is recommended for large datasets
* Softmax scores are relative similarity rankings

---

# Example Integration

```python
from test import encode_text, encode_image

text_emb = encode_text(["a photo of a dog"])
image_emb = encode_image("image.jpg")

scores = (image_emb @ text_emb.T)[0] * 100

print(scores)
```

---

# Credits

* OpenAI CLIP
* Hugging Face Transformers
* ONNX Runtime
