import os
import glob
import numpy as np
import onnxruntime as ort
from transformers import CLIPTokenizer
from PIL import Image

MODEL = "openai/clip-vit-base-patch32"
MAX_LEN = 77
LOGIT_SCALE = 100.0

print("Loading ONNX models...")
tokenizer = CLIPTokenizer.from_pretrained(MODEL)

# Use FP32 text model — INT8 was degrading embeddings
text_sess = ort.InferenceSession("clip_text.onnx")
image_sess = ort.InferenceSession("clip_image.onnx")

def preprocess_image(path):
    img = Image.open(path).convert("RGB").resize((224, 224))
    arr = np.array(img).astype("float32") / np.float32(255.0)
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))
    return arr[np.newaxis, :]

def normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)

def encode_text(prompts):
    inputs = tokenizer(prompts, return_tensors="np", padding="max_length", truncation=True, max_length=MAX_LEN)
    out = text_sess.run(None, {"input_ids": inputs["input_ids"]})[0]
    return normalize(out)

def encode_image(path):
    img = preprocess_image(path)
    out = image_sess.run(None, {"pixel_values": img})[0]
    return normalize(out)

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

if __name__ == "__main__":
    # Proper CLIP zero-shot: compare image against competing labels
    # The model picks the best match from this list
    candidate_prompts = [
    "a golden retriever puppy sitting on grass",
    "a black ford mustang sports car",
]

    print(f"\nCandidates: {candidate_prompts}")
    text_embs = encode_text(candidate_prompts)  # shape: (N_labels, 512)

    image_files = glob.glob("*.jpg") + glob.glob("*.jpeg") + glob.glob("*.png")
    # Exclude the ONNX-unrelated files just in case
    image_files = [f for f in image_files if not f.endswith(".onnx")]

    if len(image_files) == 0:
        print("No images found!")
        exit()

    print(f"\nFound {len(image_files)} images.\n")
    print("=" * 60)

    with open("results.txt", "w", encoding="utf-8") as f:
        for img_path in image_files:
            try:
                image_emb = encode_image(img_path)  # shape: (1, 512)
                logits = (image_emb @ text_embs.T)[0] * LOGIT_SCALE  # shape: (N_labels,)
                probs = softmax(logits)

                best_idx = int(np.argmax(probs))
                best_label = candidate_prompts[best_idx]
                best_prob = probs[best_idx]

                header = f"Image: {img_path}"
                print(header)
                f.write(header + "\n")

                for label, prob in zip(candidate_prompts, probs):
                    marker = " <-- BEST" if label == best_label else ""
                    line = f"  {prob*100:6.2f}%  {label}{marker}"
                    print(line)
                    f.write(line + "\n")

                print()
                f.write("\n")

            except Exception as e:
                print(f"Failed to process {img_path}: {e}")

    print("=" * 60)
    print("\n✅ Results saved to results.txt")