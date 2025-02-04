"""MambaLM model for onnx conversion."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields

import torch
from torch import nn
from transformers.utils import CONFIG_NAME, WEIGHTS_NAME
from transformers.utils.hub import cached_file

from mamba2onnx.mamba import Config, Mamba, RMSNorm


@dataclass
class MambaLMConfig(Config):
    vocab_size: int
    pad_vocab_size_multiple: int

    def __post_init__(self):
        super().__post_init__()

    def to_mamba_config(self) -> Config:
        mamba_config_fields = {field.name for field in fields(Config)}
        return Config(**{key: value for key, value in asdict(self).items() if key in mamba_config_fields})


class MambaLM(nn.Module):
    def __init__(self, config: MambaLMConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.mamba = Mamba(self.config.to_mamba_config())
        self.norm = RMSNorm(self.config.d_model)
        self.lm_head = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def init_caches(self) -> tuple[torch.Tensor, torch.Tensor]:
        hs = torch.zeros(self.config.n_layers, 1, self.config.d_inner, self.config.d_state)
        inputs = torch.zeros(self.config.n_layers, 1, self.config.d_inner, self.config.d_conv - 1)
        return hs, inputs

    def forward(
        self, token: torch.Tensor, hs: torch.Tensor, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        x = self.embedding(token)
        x, hs, inputs = self.mamba.step(x, hs, inputs)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, (hs, inputs)

    @staticmethod
    def from_pretrained(name: str) -> MambaLM:
        def load_config_hf(model_name: str) -> dict:
            resolved_archive_file = cached_file(model_name, CONFIG_NAME, _raise_exceptions_for_missing_entries=False)
            return json.load(open(resolved_archive_file))  # pyright: ignore[reportArgumentType]

        def load_state_dict_hf(model_name: str) -> dict:
            resolved_archive_file = cached_file(model_name, WEIGHTS_NAME, _raise_exceptions_for_missing_entries=False)
            return torch.load(resolved_archive_file, weights_only=True, map_location="cpu", mmap=True)  # pyright: ignore[reportArgumentType]

        # copy config data
        config_data = load_config_hf(name)
        config = MambaLMConfig(
            d_model=config_data["d_model"],
            d_delta=math.ceil(config_data["d_model"] / 16),
            d_conv=4,
            d_state=16,
            n_layers=config_data["n_layer"],
            vocab_size=config_data["vocab_size"] + 3,
            pad_vocab_size_multiple=config_data["pad_vocab_size_multiple"],
        )

        model = MambaLM(config)

        # copy weights
        state_dict = load_state_dict_hf(name)

        # I modified structure of the model for better readability, so I need to re-organize the weights
        new_state_dict = {}
        for key in state_dict:
            if key in ("backbone.embedding.weight", "backbone.norm_f.weight"):
                new_key = key.replace("backbone.", "")
            else:
                new_key = key.replace("backbone", "mamba")

            if "conv1d" in new_key:
                new_key = new_key.replace("conv1d", "conv")

            if "mixer" in new_key:
                new_key = new_key.replace("mixer", "block")

            if "norm.weight" in new_key:
                new_key = new_key.replace("norm.weight", "norm.weights")
            if "norm_f.weight" in new_key:
                new_key = new_key.replace("norm_f.weight", "norm.weights")

            ssm_blocks = ["A_log", "D", "x_proj", "dt_proj"]
            for block in ssm_blocks:
                if f"block.{block}" in new_key:
                    new_key = new_key.replace(block, f"ssm.{block}")

            new_state_dict[new_key] = state_dict[key]

        # pprint(new_state_dict.keys())
        model.load_state_dict(new_state_dict)

        return model
