# CLIP-ONNX

```markdown
# ONNX CLIP Image-Text Similarity

This project demonstrates how to export OpenAI's CLIP (ViT-B/32) model to ONNX format, apply INT8 quantization to the text model for faster inference, and calculate image-text similarity using `onnxruntime`.

## 📁 Project Structure

.
├── export_clip.py      # Exports the CLIP text encoder to ONNX
├── export_image.py     # Exports the CLIP vision encoder to ONNX
├── quantize.py         # Dynamically quantizes the text model to INT8
├── test_onnx.py        # Quick test script for text embeddings
├── clip_full.py        # Final script: compares an image to a text prompt
├── test.jpg            # (Required) Sample image for testing
└── README.md           # This file
```

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.8 or higher
- Git

### 2. Clone the Repository

### 3. Create and Activate a Virtual Environment

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install torch transformers onnx onnxruntime numpy Pillow
```

## 🛠️ Usage & Pipeline

Follow these steps in order to generate the ONNX models and test the similarity.

### Step 1: Export the Base ONNX Models
Export the text and image encoders from the Hugging Face Transformers library.

```bash
python export_clip.py
# Output: clip_text.onnx

python export_image.py
# Output: clip_image.onnx
```

### Step 2: Quantize the Text Model (Optional but Recommended)
This reduces the file size and speeds up inference by converting the text model weights to INT8.

```bash
python quantize.py
# Output: clip_text_int8.onnx
```

### Step 3: Test Text Embeddings
Verify that the text ONNX model is working and outputs normalized embeddings. *(Note: This script uses the unquantized model).*

```bash
python test_onnx.py
```

### Step 4: Run Full Image-Text Similarity
This script uses the quantized text model (`clip_text_int8.onnx`) and the image model (`clip_image.onnx`) to calculate the cosine similarity between a detailed text prompt and an image.

1. Ensure you have an image named `test.jpg` in the root directory.
2. Run the script:

```bash
python clip_full.py
```
*You should see an output like: `Similarity: 0.284...` (higher numbers indicate a closer match).*

