"""The script loads pretrained MambaLM and converts the MambaLM model to ONNX format."""

import torch

from mamba2onnx.mamba_lm import MambaLM

model = MambaLM.from_pretrained("state-spaces/mamba-130m")
model.eval()

torch.onnx.export(
    model,
    (torch.zeros(1, dtype=torch.int64), *model.init_caches()),
    "mamba.onnx",
    input_names=["token", "hs", "inputs"],
    output_names=["logits", "hs", "inputs"],
    opset_version=17,
)
