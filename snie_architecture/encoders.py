from __future__ import annotations

from typing import Protocol

import torch
import torch.nn as nn


class TokenEncoder(Protocol):
    """Encoder contract expected by SNIEModel.

    Implementations should return:
    - tokens: shape (batch, time, encoder_dim)
    - pooled: shape (batch, encoder_dim)
    """

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ...


class FeatureTokenEncoder(nn.Module):
    """Generic token encoder for precomputed frame or segment features.

    This module is included only as a public, data-agnostic example encoder. For
    audio, text, image, or multimodal inputs, replace it with any encoder that
    follows the TokenEncoder contract.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        depth: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")

        layers: list[nn.Module] = [nn.Linear(input_dim, output_dim), nn.GELU()]
        for _ in range(depth - 1):
            layers.extend(
                [
                    nn.LayerNorm(output_dim),
                    nn.Dropout(dropout),
                    nn.Linear(output_dim, output_dim),
                    nn.GELU(),
                ]
            )
        self.net = nn.Sequential(*layers)
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError("FeatureTokenEncoder expects input shape (batch, time, feature_dim)")
        tokens = self.net(x)
        pooled = tokens.mean(dim=1)
        return tokens, pooled
