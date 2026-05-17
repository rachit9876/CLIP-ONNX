# CLIP-ONNX

Run OpenAI CLIP locally using ONNX Runtime on CPU, integrated GPUs, and low power devices.

Lightweight, offline, and optimized for fast semantic image inference without CUDA.

---

## Features

* Offline CLIP inference
* ONNX Runtime backend
* CPU + iGPU optimized
* Zero shot image classification
* Semantic image search
* Cross platform support
* Lightweight local AI pipeline

---

## Quick Start

### Clone Repository

```bash id="b9j3do"
git clone https://github.com/rachit9876/CLIP-ONNX.git
cd CLIP-ONNX
```

### Create Virtual Environment

#### Windows

```bash id="r9zz2o"
python -m venv venv
venv\Scripts\activate
```

#### Linux / macOS

```bash id="0e5w2t"
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash id="z3fhd7"
pip install torch transformers onnxruntime onnx Pillow numpy
```

### Build Models

```bash id="2u10pw"
python build.py
```

### Run Inference

```bash id="qz5m8d"
python test.py
```

---

## Example

```python id="k2zll9"
candidate_prompts = [
    "a photo of a dog",
    "a sports car",
    "a cat",
    "a person",
]
```

### Output

```text id="plh7c2"
Image: dog.jpg

82.31%  a photo of a dog
7.14%   a cat
5.02%   a person
2.91%   a sports car
```

---

## Repository Structure

```text id="0j6s2k"
build.py         Export CLIP models to ONNX
test.py          Run similarity inference
clip_image.onnx  Vision encoder
clip_text.onnx   Text encoder
```

---

## Supported Models

Default:

```python id="xwpfq8"
MODEL = "openai/clip-vit-base-patch32"
```

Compatible:

```text id="y5m95x"
openai/clip-vit-base-patch16
openai/clip-vit-large-patch14
```

---

## Supported Hardware

Designed for:

* Intel Iris Xe
* AMD integrated graphics
* CPU only systems
* Thin laptops
* Edge devices
* Mini PCs

No CUDA required.

---

## Performance Notes

* Images resized to `224×224`
* FP32 preserves embedding quality
* Batched inference recommended for large datasets
* Softmax scores are relative similarity rankings

---

## Example Integration

```python id="l5u7o7"
from test import encode_text, encode_image

text_emb = encode_text(["a photo of a dog"])
image_emb = encode_image("image.jpg")

scores = (image_emb @ text_emb.T)[0] * 100

print(scores)
```

---

## Credits

* OpenAI CLIP
* Hugging Face Transformers
* ONNX Runtime
