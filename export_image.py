import torch
from torch import nn
from transformers import CLIPModel

MODEL = "openai/clip-vit-base-patch32"

class SafeVision(nn.Module):
    def __init__(self):
        super().__init__()
        base = CLIPModel.from_pretrained(MODEL)
        self.embeddings = base.vision_model.embeddings
        self.encoder = base.vision_model.encoder
        self.post_layernorm = base.vision_model.post_layernorm
        self.visual_projection = base.visual_projection

    def forward(self, pixel_values):
        x = self.embeddings(pixel_values)
        x = self.encoder(inputs_embeds=x).last_hidden_state
        x = self.post_layernorm(x)
        x = x[:, 0, :]
        x = self.visual_projection(x)
        return x

model = SafeVision().eval()
dummy = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy,
    "clip_image.onnx",
    input_names=["pixel_values"],
    output_names=["image_emb"],
    dynamic_axes={
        "pixel_values": {0: "batch"},
        "image_emb": {0: "batch"},
    },
    opset_version=18,
)

print("IMAGE MODEL EXPORTED")
