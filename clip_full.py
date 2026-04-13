import numpy as np
import onnxruntime as ort
from transformers import CLIPTokenizer
from PIL import Image

MODEL = "openai/clip-vit-base-patch32"
MAX_LEN = 16

# -------- tokenizer --------
tokenizer = CLIPTokenizer.from_pretrained(MODEL)

# -------- sessions --------
text_sess = ort.InferenceSession("clip_text_int8.onnx")
image_sess = ort.InferenceSession("clip_image.onnx")

# -------- preprocess --------
def preprocess_image(path):
    from PIL import Image
    import numpy as np

    img = Image.open(path).convert("RGB")
    img = img.resize((224, 224))

    arr = np.array(img).astype("float32")
    arr = arr / np.float32(255.0)

    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))

    return arr[np.newaxis, :]

# -------- text embedding --------
def encode_text(texts):
    inputs = tokenizer(
        texts,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
    )

    out = text_sess.run(
        None,
        {
            "input_ids": inputs["input_ids"], # <-- ONLY input_ids, nothing else
        },
    )[0]

    return normalize(out)
# -------- image embedding --------
def encode_image(path):
    img = preprocess_image(path)

    out = image_sess.run(
        None,
        {"pixel_values": img},
    )[0]

    return normalize(out)

# -------- normalize --------
def normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)

# -------- similarity --------
# -------- similarity --------
def similarity(a, b):
    return float((a @ b.T)[0, 0])  # <-- Added [0, 0] to extract the scalar

# -------- test --------
if __name__ == "__main__":
    # 试试完整的句子描述
    text = encode_text(["A light cream-colored Golden Retriever puppy is lying on fresh green grass outdoors, centered in the frame and facing forward at eye level. The puppy has soft, fluffy fur with slightly wavy texture, especially around its floppy ears and neck, while the snout appears smoother. Its dark brown eyes are round and reflective, giving an alert yet gentle expression, and its black nose is moist and well-defined. The mouth is slightly open with a hint of tongue visible, creating a friendly, almost smiling appearance. The puppy’s posture is relaxed, with its body resting on the grass and head held upright. The foreground grass is detailed and vibrant, while the background is heavily blurred with dark green tones, likely trees or bushes, producing a strong bokeh effect that isolates the subject. Lighting is natural and soft, evenly illuminating the puppy without harsh shadows, enhancing the warm tones of its coat and the overall calm, inviting atmosphere of the scene."])
    image = encode_image("test.jpg")

    print("Similarity:", similarity(text, image))
