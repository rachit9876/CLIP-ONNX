import numpy as np
import onnxruntime as ort
from transformers import CLIPTokenizer

MODEL = "openai/clip-vit-base-patch32"
MAX_LEN = 16

tokenizer = CLIPTokenizer.from_pretrained(MODEL)

texts = ["a dog", "a cat", "a car"]

inputs = tokenizer(
    texts,
    return_tensors="np",
    padding="max_length",
    truncation=True,
    max_length=MAX_LEN,
)

session = ort.InferenceSession("clip_text.onnx")

outputs = session.run(
    None,
    {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
    },
)

embeddings = outputs[0]

# normalize
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

print("Embeddings shape:", embeddings.shape)
print(embeddings)
