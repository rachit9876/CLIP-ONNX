import torch
from torch import nn
from transformers import CLIPModel, CLIPConfig

MODEL = "openai/clip-vit-base-patch32"
MAX_LEN = 16

class SafeCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        base = CLIPModel.from_pretrained(MODEL)
        config = CLIPConfig.from_pretrained(MODEL)

        self.embeddings = base.text_model.embeddings
        self.encoder = base.text_model.encoder
        self.final_layer_norm = base.text_model.final_layer_norm
        self.text_projection = base.text_projection

        self.eos_token_id = config.text_config.eos_token_id

    @torch.no_grad() # <-- Added to prevent tracer warnings
    def forward(self, input_ids):
        x = self.embeddings(input_ids=input_ids)

        # is_causal=False stops the PyTorch tracer warning about Python booleans
        x = self.encoder(inputs_embeds=x, is_causal=False).last_hidden_state
        x = self.final_layer_norm(x)

        eos_index = (input_ids == self.eos_token_id).int().argmax(dim=-1)
        x = x[torch.arange(x.shape[0]), eos_index]

        x = self.text_projection(x)
        return x

model = SafeCLIP().eval()
dummy_input_ids = torch.ones((1, MAX_LEN), dtype=torch.long)

torch.onnx.export(
    model,
    dummy_input_ids,
    "clip_text.onnx",
    input_names=["input_ids"],
    output_names=["text_emb"],
    dynamic_axes={
        "input_ids": {0: "batch"},
        "text_emb": {0: "batch"},
    },
    opset_version=18,
    dynamo=False
)

print("DONE CLEAN EXPORT (REAL CLIP)")
