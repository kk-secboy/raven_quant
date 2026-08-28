"""Platform-owned CPU sequence models used by the independent Qlib runner.

This module is copied into the read-only model sandbox.  It deliberately only
contains model definitions: training, early stopping, data loading and scoring
remain owned by Qlib's ``GeneralPTNN`` adapter in ``model_sandbox_runner.py``.
The architectures are fixed so an RD-Agent candidate cannot silently enlarge a
screening job or switch to CUDA.
"""

from __future__ import annotations

import math

import torch
from torch import nn

TEMPLATE_CONTRACT_VERSION = "platform-cpu-sequence-models-v1"
SEQUENCE_LENGTH = 20


class GovernedGRU(nn.Module):
    """One-layer 32-unit GRU with an explicit output dropout.

    PyTorch ignores the built-in recurrent ``dropout`` argument when
    ``num_layers == 1``.  The recurrent dropout is therefore fixed to zero and
    the governed 0.1 dropout is applied explicitly to the final hidden state.
    """

    def __init__(
        self,
        *,
        num_features: int,
        num_timesteps: int = SEQUENCE_LENGTH,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_features < 1 or num_timesteps != SEQUENCE_LENGTH:
            raise ValueError("governed GRU requires positive features and a 20-day sequence")
        if hidden_size != 32 or num_layers != 1 or float(dropout) != 0.1:
            raise ValueError("governed GRU architecture is immutable")
        self.num_features = int(num_features)
        self.num_timesteps = int(num_timesteps)
        self.gru = nn.GRU(
            input_size=self.num_features,
            hidden_size=32,
            num_layers=1,
            dropout=0.0,
            batch_first=True,
        )
        self.output_dropout = nn.Dropout(p=0.1)
        self.output = nn.Linear(32, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3:
            raise ValueError("governed GRU expects [batch, day, feature]")
        if values.shape[1] != self.num_timesteps or values.shape[2] != self.num_features:
            raise ValueError("governed GRU input does not match the frozen data contract")
        encoded, _ = self.gru(values)
        return self.output(self.output_dropout(encoded[:, -1, :]))


class _SinusoidalPosition(nn.Module):
    def __init__(self, *, d_model: int, length: int) -> None:
        super().__init__()
        position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / d_model)
        )
        encoding = torch.zeros(length, d_model, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=True)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.encoding[:, : values.shape[1], :]


class GovernedTransformer(nn.Module):
    """Small two-layer CPU Transformer for the 20-day sequence lane."""

    def __init__(
        self,
        *,
        num_features: int,
        num_timesteps: int = SEQUENCE_LENGTH,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_features < 1 or num_timesteps != SEQUENCE_LENGTH:
            raise ValueError(
                "governed Transformer requires positive features and a 20-day sequence"
            )
        if d_model != 32 or nhead != 4 or num_layers != 2 or float(dropout) != 0.1:
            raise ValueError("governed Transformer architecture is immutable")
        self.num_features = int(num_features)
        self.num_timesteps = int(num_timesteps)
        self.input_projection = nn.Linear(self.num_features, 32)
        self.position = _SinusoidalPosition(d_model=32, length=SEQUENCE_LENGTH)
        layer = nn.TransformerEncoderLayer(
            d_model=32,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.output_norm = nn.LayerNorm(32)
        self.output = nn.Linear(32, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3:
            raise ValueError("governed Transformer expects [batch, day, feature]")
        if values.shape[1] != self.num_timesteps or values.shape[2] != self.num_features:
            raise ValueError(
                "governed Transformer input does not match the frozen data contract"
            )
        encoded = self.encoder(self.position(self.input_projection(values)))
        return self.output(self.output_norm(encoded[:, -1, :]))


__all__ = [
    "GovernedGRU",
    "GovernedTransformer",
    "SEQUENCE_LENGTH",
    "TEMPLATE_CONTRACT_VERSION",
]
