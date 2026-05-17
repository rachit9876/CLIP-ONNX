import torch
from torch import nn
from transformers import CLIPModel, CLIPConfig
from onnxruntime.quantization import quantize_dynamic, QuantType
import os

MODEL = "openai/clip-vit-base-patch32"
MAX_LEN = 77

# -------- 1. TEXT ENCODER --------
class SafeCLIP(nn.Module):
    def __init__(self, base, config):
        super().__init__()
        self.embeddings = base.text_model.embeddings
        self.encoder = base.text_model.encoder
        self.final_layer_norm = base.text_model.final_layer_norm
        self.text_projection = base.text_projection
        self.eos_token_id = config.text_config.eos_token_id

    @torch.no_grad()
    def forward(self, input_ids):
        x = self.embeddings(input_ids=input_ids)
        x = self.encoder(inputs_embeds=x, is_causal=False).last_hidden_state
        x = self.final_layer_norm(x)
        eos_index = (input_ids == self.eos_token_id).int().argmax(dim=-1)
        x = x[torch.arange(x.shape[0]), eos_index]
        return self.text_projection(x)

# -------- 2. VISION ENCODER --------
class SafeVision(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.embeddings = base.vision_model.embeddings
        self.pre_layrnorm = base.vision_model.pre_layrnorm  # ← was missing, this is the bug
        self.encoder = base.vision_model.encoder
        self.post_layernorm = base.vision_model.post_layernorm
        self.visual_projection = base.visual_projection

    @torch.no_grad()
    def forward(self, pixel_values):
        x = self.embeddings(pixel_values)
        x = self.pre_layrnorm(x)               # ← normalize BEFORE encoder
        x = self.encoder(inputs_embeds=x).last_hidden_state
        x = self.post_layernorm(x)
        x = x[:, 0, :]
        return self.visual_projection(x)

if __name__ == "__main__":
    print("Loading Hugging Face model...")
    base_model = CLIPModel.from_pretrained(MODEL)
    config = CLIPConfig.from_pretrained(MODEL)

    # --- TEXT PIPELINE ---
    print("\n--- Exporting Text Model ---")
    text_model = SafeCLIP(base_model, config).eval()
    dummy_input_ids = torch.ones((1, MAX_LEN), dtype=torch.long)
    torch.onnx.export(
        text_model, dummy_input_ids, "clip_text.onnx",
        input_names=["input_ids"], output_names=["text_emb"],
        dynamic_axes={"input_ids": {0: "batch"}, "text_emb": {0: "batch"}},
        opset_version=18, dynamo=False
    )

    print("\n--- Quantizing Text Model to INT8 ---")
    quantize_dynamic("clip_text.onnx", "clip_text_int8.onnx", weight_type=QuantType.QInt8)

    # --- IMAGE PIPELINE (FULL PRECISION) ---
    print("\n--- Exporting Image Model ---")
    image_model = SafeVision(base_model).eval()
    dummy_pixel = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        image_model, dummy_pixel, "clip_image.onnx",
        input_names=["pixel_values"], output_names=["image_emb"],
        dynamic_axes={"pixel_values": {0: "batch"}, "image_emb": {0: "batch"}},
        opset_version=18,
        dynamo=False
    )

    # --- CLEANUP ---
    print("\n--- Cleaning up ---")
    # clip_text.onnx kept intentionally (FP32 needed, INT8 degrades embeddings)
    # if os.path.exists("clip_text.onnx"): os.remove("clip_text.onnx")

    print("\n✅ Done! Models: clip_text.onnx, clip_text_int8.onnx, clip_image.onnx")