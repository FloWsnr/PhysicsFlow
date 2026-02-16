"""Embedding utilities for generative models.

This module provides time embeddings and conditioning projections
for flow matching models.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor


def sinusoidal_embedding(t: Tensor, dim: int, max_period: float = 10000.0) -> Tensor:
    """Create sinusoidal positional embeddings for time values.

    This is the standard positional embedding used in transformers.

    Parameters
    ----------
    t : Tensor
        Time values, shape (batch_size,) or (batch_size, 1).
        Values are typically in [0, 1] for flow matching.
    dim : int
        Embedding dimension (should be even).
    max_period : float, optional
        Maximum period for the sinusoidal functions, by default 10000.0.

    Returns
    -------
    Tensor
        Embeddings of shape (batch_size, dim).
    """
    if t.dim() == 1:
        t = t.unsqueeze(-1)

    half_dim = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half_dim, device=t.device, dtype=t.dtype)
        / half_dim
    )
    args = t * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TimeEmbedding(nn.Module):
    """Learnable time embedding layer.

    Combines sinusoidal embedding with an MLP for richer representations.

    Parameters
    ----------
    dim : int
        Output embedding dimension.
    hidden_dim : int, optional
        Hidden dimension for MLP. Default: 4 * dim.
    """

    def __init__(self, dim: int, hidden_dim: int | None = None):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}.")
        if dim % 2 != 0:
            raise ValueError(
                f"dim must be even for sinusoidal_embedding, got {dim}."
            )
        self.dim = dim
        hidden_dim = hidden_dim or 4 * dim

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        """Compute time embedding.

        Parameters
        ----------
        t : Tensor
            Time values, shape (batch_size,).

        Returns
        -------
        Tensor
            Embeddings of shape (batch_size, dim).
        """
        emb = sinusoidal_embedding(t, self.dim)
        return self.mlp(emb)


class ConditioningProjection(nn.Module):
    """Project conditioning vectors (physics parameters) to embedding space.

    Parameters
    ----------
    input_dim : int
        Dimension of input conditioning vector.
    output_dim : int
        Dimension of output embedding.
    hidden_dim : int, optional
        Hidden layer dimension. Default: output_dim.
    """

    def __init__(
        self, input_dim: int, output_dim: int, hidden_dim: int | None = None
    ):
        super().__init__()
        hidden_dim = hidden_dim or output_dim

        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, c: Tensor) -> Tensor:
        """Project conditioning to embedding space.

        Parameters
        ----------
        c : Tensor
            Conditioning vector, shape (batch_size, input_dim).

        Returns
        -------
        Tensor
            Projected embedding, shape (batch_size, output_dim).
        """
        return self.proj(c)
