"""Simple Mamba Implmentation in PyTorch.

This is a simple implementation of Mamba for understanding the Mamba algorithm.
This file closely follows the mamba_simple.py from the official Mamba implementation.
The major diffeernces are:
- Exclude CUDA-only operation
- The selective scan is done using PyTorch.
- Doesn't use the einops library.
- Some configurations are omitted for simplicity.
"""

from dataclasses import asdict, dataclass, fields

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    # For Mamba
    n_layers: int
    d_conv: int
    # For SSM
    d_model: int
    d_state: int  # N in paper
    d_delta: int

    # For discretization
    dt_min: float
    dt_max: float
    dt_init: float
    dt_scale: float
    dt_init_floor: float

    def __post_init__(self):
        self.d_inner = self.d_model * 2  # D in paper


class Mamba(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([ResidualBlock(config) for _ in range(config.n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, d_model] -> [B, L, d_model]
        for layer in self.layers:
            x = layer(x)
        return x

    def step(
        self, x: torch.Tensor, hs: torch.Tensor, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, L, d_model), hs: [n_layers, B, D, N], inputs: [n_layers, B, D, d_conv-1] -> [B, L, d_model], hs, inputs
        for i, layer in enumerate(self.layers):
            x, hs[i], inputs[i] = layer.step(x, hs[i], inputs[i])  # pyright: ignore[reportCallIssue]
        return x, hs, inputs


class ResidualBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.block = MambaBlock(config)
        self.norm = RMSNorm(config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, d_model] -> [B, L, d_model]
        return self.block(self.norm(x)) + x

    def step(
        self, x: torch.Tensor, hs: torch.Tensor, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, L, d_model], hs: [B, D, N], inputs: [B, D, d_conv-1] -> [B, L, d_model], hs, inputs
        out, cache = self.block.step(self.norm(x), (hs, inputs))
        return out + x, cache[0], cache[1]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class MambaBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.in_proj = nn.Linear(
            config.d_model,
            2 * config.d_inner,
            bias=False,
        )

        self.conv = nn.Conv1d(
            in_channels=config.d_inner,
            out_channels=config.d_inner,
            kernel_size=config.d_conv,
            bias=True,
            groups=config.d_inner,
            padding=config.d_conv - 1,
        )
        self.ssm = SSM(config)
        self.out_proj = nn.Linear(
            config.d_inner,
            config.d_model,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, d_model] -> output: [B, L, d_model]
        batch_size, seq_len, _ = x.shape

        # x: [B, L, D] -> xz[B, L, 2*D]
        xz = self.in_proj(x)
        # xz: [B, L, 2*D] -> x, z: [B, L, D]
        x, z = xz.chunk(2, dim=-1)

        x = x.transpose(1, 2)
        x = self.conv(x)[:, :, :, seq_len]
        x = x.transpose(1, 2)

        x = F.silu(x)
        y = self.ssm(x, z)

        z = F.silu(z)

        output = y * z
        output = self.out_proj(output)
        return output

    def step(
        self, x: torch.Tensor, cache: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Inference step.

        The cache stores two things for each layer:
        - The hidden state h [B, D, N]
        - The last d_conv-1 inputs [B, D, d_conv-1]

        Initially, the cache should be (None, torch.zeros())
        """
        h, inputs = cache

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=1)

        x_cache = x.unsqueeze(2)
        x = self.conv(torch.cat([inputs, x_cache], dim=2))[:, :, self.config.d_conv - 1]

        x = F.silu(x)
        y, h = self.ssm.step(x, h)

        z = F.silu(z)

        output = y * z
        output = self.out_proj(output)

        # Prepare the cache for the next step
        inputs = torch.cat([inputs[:, :, 1:], x_cache], dim=2)
        cache = (h, inputs)
        return output, cache


class SSM(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # A, A_log: [D, N]
        # Store A_log instead of A for following reasons:
        # 1. Being strictly negative A (A= -exp(A_log))
        # 2. Numerical stability during training.
        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))

        # D: [D]
        self.D = nn.Parameter(torch.ones(config.d_inner))

        # x_proj: [D,  + 2 * D_state]
        # x_proj projects x -> delta, B, C
        self.x_proj = nn.Linear(config.d_inner, config.d_delta + 2 * config.d_state, bias=False)

        # dt_proj: [d_delta, D]
        # dt_proj projects delta from d_delta -> D
        self.dt_proj = nn.Linear(config.d_delta, config.d_inner, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D], z: [B, L, D] -> output: [B, L, D]
        A = -torch.exp(self.A_log)
        D = self.D.float()

        delta, B, C = torch.split(
            self.x_proj(x), [self.config.d_delta, self.config.d_state, self.config.d_state], dim=-1
        )
        discretized_A, discretized_B = self._discretization(delta, A, B)
        return self._selective_scan(x, discretized_A, discretized_B, C, D)

    def _discretization(
        self,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Discretization of the continuous-time system."""
        # discretization step
        # [B, L, d_delta] -> [B, L, D]
        delta = F.softplus(self.dt_proj(delta))

        # Zero Order Hold (ZOH) discretization
        discretized_A = torch.exp(delta.unsqueeze(-1) * A)

        # Simplified Euler discretization instead of ZOH
        # Performance is similar to ZOH, but faster
        discretized_B = delta.unsqueeze(-1) * B.unsqueeze(2)
        return discretized_A, discretized_B

    def _selective_scan(
        self,
        x: torch.Tensor,
        A_bar: torch.Tensor,
        B_bar: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
    ) -> torch.Tensor:
        """Parallel selective scan."""
        # Input shapes:
        # x: [B, L, D]
        # A_bar: [B, L, D, N]
        # B_bar: [B, L, D, N]
        # C: [B, L, N]
        # D: [D]
        # y: [B, L, D]
        batch_size, seq_len, _ = x.shape

        # Bx: [B, L, D, N]
        Bx = B_bar * (x.unsqueeze(-1))

        # Scan
        h = torch.zeros(batch_size, self.config.d_inner, self.config.d_state, device=A_bar.device)  # [B, D, N]
        hs = []
        for t in range(seq_len):
            h = A_bar[:, t] * h + Bx[:, t]
            hs.append(h)
        hs = torch.stack(hs, dim=1)

        # [B, L, D, N] @ [B, L, N, 1] -> [B, L, D, 1]
        y = (hs @ C.unsqueeze(-1)).squeeze(3)
        return y + D * x

    def step(self, x: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Inference step."""
        batch_size, _, _ = x.shape
        A = -torch.exp(self.A_log)
        D = self.D.float()

        deltaBC = self.x_proj(x)
        delta, B, C = torch.split(deltaBC, [self.config.d_delta, self.config.d_state, self.config.d_state], dim=-1)
        discretized_A, discretized_B = self._discretization(delta, A, B)

        Bx = discretized_B * x.unsqueeze(-1)
        if h is None:
            h = torch.zeros(batch_size, self.config.d_inner, self.config.d_state, device=A.device)
        # h: [B, D, N]
        h = discretized_A * h + Bx

        # [B, D, N] @ [B, N, 1] -> [B, D, 1]
        y = (h @ C.unsqueeze(-1)).squeeze(2)
        y = y + D * x
        return y, h
