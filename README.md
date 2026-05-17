# CLIP-ONNX

Run OpenAI's CLIP model fully offline using ONNX Runtime. No GPU required : built for systems with iGPUs.

Given a text prompt and a folder of images, it ranks how well each image matches the prompt.

---

## How It Works

CLIP (ViT-B/32) has two encoders — one for text, one for images. Both produce embeddings in the same vector space, so you can measure similarity between them using cosine similarity.

This project exports both encoders to ONNX format for fast CPU inference, then uses softmax over a set of candidate prompts to classify each image.

```
Image → Vision Encoder (FP32) → image embedding (512-dim)
Text  → Text Encoder  (FP32) → text embedding  (512-dim)
                                      ↓
                         cosine similarity × logit_scale
                                      ↓
                              softmax → % match
```

---

## Files

| File | Purpose |
|------|---------|
| `build.py` | Downloads CLIP from HuggingFace, exports to ONNX |
| `test.py` | Runs inference on images in the current folder |
| `clip_image.onnx` | Vision encoder — FP32, ~351MB |
| `clip_text.onnx` | Text encoder — FP32, ~254MB |

> **Note:** `clip_text_int8.onnx` is also generated during build but not used. INT8 quantization degraded embedding quality enough to flip rankings. Delete it after building.

---

## Setup

**Requirements**

```
Python 3.10+
torch
transformers
onnxruntime
onnx
Pillow
numpy
```

Install:

```bash
pip install torch transformers onnxruntime onnx Pillow numpy
```

**Build the ONNX models** (one-time, needs internet to download CLIP weights ~600MB):

```bash
python build.py
```

This generates `clip_image.onnx` and `clip_text.onnx` in the current folder.

---

## Usage

1. Put your images (`.jpg`, `.jpeg`, `.png`) in the same folder as `test.py`
2. Open `test.py` and edit the candidate prompts at the top of `__main__`:

```python
candidate_prompts = [
    "a photo of a golden retriever puppy",
    "a black sports car parked outdoors",
    "a photo of a cat",
    "a photo of a person",
    "a photo of a landscape",
]
```

3. Run:

```bash
python test.py
```

**Example output:**

```
Image: test_dog.jpg
   22.32%  a photo of a golden retriever puppy  <-- BEST
   19.12%  a photo of a car
   19.69%  a photo of a cat
   19.71%  a photo of a person
   19.16%  a photo of a landscape
```

Results are also saved to `results.txt`.

---

## Writing Good Prompts

CLIP was trained on image captions, so natural caption-style prompts work best.

| Instead of | Use |
|------------|-----|
| `"dog"` | `"a photo of a dog"` |
| `"car"` | `"a black sports car parked outdoors"` |
| `"forest"` | `"a dense green forest with tall trees"` |

The more specific and descriptive, the better. Generic one-word labels give weak signal.

Also: CLIP reads the **whole image**, not just the subject. A car photo with a dramatic sky background may score higher on "landscape" than "car" if the background dominates the frame.

---

## Adapting the Code

**Change the model**

In both `build.py` and `test.py`, update:

```python
MODEL = "openai/clip-vit-base-patch32"
```

Other supported models: `openai/clip-vit-large-patch14`, `openai/clip-vit-base-patch16`
Larger models = better accuracy, bigger ONNX files.

**Scan a specific folder instead of current directory**

In `test.py`, replace:

```python
image_files = glob.glob("*.jpg") + glob.glob("*.jpeg") + glob.glob("*.png")
```

With:

```python
folder = r"C:\path\to\your\images"
image_files = glob.glob(f"{folder}/*.jpg") + glob.glob(f"{folder}/*.jpeg") + glob.glob(f"{folder}/*.png")
```

**Use it as a module in your own code**

```python
from test import encode_text, encode_image, similarity, softmax
import numpy as np

text_emb = encode_text(["a photo of a dog"])
image_emb = encode_image("myimage.jpg")

logits = (image_emb @ text_emb.T)[0] * 100.0
print(f"Score: {logits[0]:.2f}")
```

---

## Known Limitations

- Images are resized to 224×224 before inference. Very wide or tall images will be squished — crop first for best results.
- Softmax scores are relative to your candidate list. A "best" pick at 22% just means it beat your other options — it's not a confidence score in the absolute sense.
- CPU inference on a large image folder is slow. For batches over ~500 images, consider batching `encode_image` calls.

---

## Credits

Model weights: [OpenAI CLIP](https://github.com/openai/CLIP)
Exported via: [HuggingFace Transformers](https://huggingface.co/openai/clip-vit-base-patch32) + [ONNX Runtime](https://onnxruntime.ai/)