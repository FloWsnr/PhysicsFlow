"""Conditioning modulation layers for DiT backbone.

This module provides adaptive layer normalization (AdaLN-Zero) and
cross-attention conditioning mechanisms for the DiT architecture.
"""

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """Apply AdaLN modulation: x * (1 + scale) + shift.

    Parameters
    ----------
    x : Tensor
        Input tensor of shape (B, N, D).
    shift : Tensor
        Shift parameter of shape (B, D) or (B, 1, D).
    scale : Tensor
        Scale parameter of shape (B, D) or (B, 1, D).

    Returns
    -------
    Tensor
        Modulated tensor of shape (B, N, D).
    """
    if shift.dim() == 2:
        shift = shift.unsqueeze(1)
    if scale.dim() == 2:
        scale = scale.unsqueeze(1)
    return x * (1 + scale) + shift


class AdaLNModulation(nn.Module):
    """Adaptive Layer Norm with Zero initialization (AdaLN-Zero).

    Produces 6 modulation parameters from conditioning embedding:
    - shift_msa, scale_msa, gate_msa for attention
    - shift_mlp, scale_mlp, gate_mlp for MLP

    Zero-initialized so the block starts as identity function.

    Parameters
    ----------
    hidden_dim : int
        Transformer hidden dimension.
    cond_embed_dim : int
        Conditioning embedding dimension.
    """

    def __init__(self, hidden_dim: int, cond_embed_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_embed_dim, 6 * hidden_dim, bias=True),
        )

        # Zero initialization for identity at start
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, c: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Compute 6 modulation parameters.

        Parameters
        ----------
        c : Tensor
            Conditioning embedding of shape (B, cond_embed_dim).

        Returns
        -------
        tuple[Tensor, ...]
            Six modulation tensors, each of shape (B, hidden_dim):
            (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        """
        params = self.adaLN_modulation(c)
        return params.chunk(6, dim=-1)


class AdaLNBlock(nn.Module):
    """Layer normalization with adaptive modulation.

    Combines LayerNorm (without affine) with AdaLN modulation.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
        """Apply normalized and modulated transformation.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (B, N, D).
        shift : Tensor
            Shift parameter of shape (B, D).
        scale : Tensor
            Scale parameter of shape (B, D).

        Returns
        -------
        Tensor
            Normalized and modulated tensor of shape (B, N, D).
        """
        return modulate(self.norm(x), shift, scale)


class CrossAttentionConditioning(nn.Module):
    """Cross-attention based conditioning.

    Alternative to AdaLN that uses cross-attention between
    sequence tokens and conditioning tokens.

    Parameters
    ----------
    hidden_dim : int
        Transformer hidden dimension.
    cond_embed_dim : int
        Conditioning embedding dimension.
    num_heads : int
        Number of attention heads.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        hidden_dim: int,
        cond_embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cond_embed_dim = cond_embed_dim

        self.norm = nn.LayerNorm(hidden_dim)

        # Project conditioning to hidden_dim if different
        self.cond_proj = (
            nn.Linear(cond_embed_dim, hidden_dim)
            if cond_embed_dim != hidden_dim
            else nn.Identity()
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """Apply cross-attention conditioning.

        Parameters
        ----------
        x : Tensor
            Sequence tokens of shape (B, N, hidden_dim).
        cond : Tensor
            Conditioning embedding of shape (B, cond_embed_dim).
            Will be expanded to (B, 1, hidden_dim) for cross-attention.

        Returns
        -------
        Tensor
            Conditioned tokens of shape (B, N, hidden_dim).
        """
        # Project and expand conditioning to (B, 1, hidden_dim)
        cond_tokens = self.cond_proj(cond)
        if cond_tokens.dim() == 2:
            cond_tokens = cond_tokens.unsqueeze(1)

        # Cross-attention: query=x, key=cond, value=cond
        x_norm = self.norm(x)
        attn_out, _ = self.cross_attn(
            query=x_norm,
            key=cond_tokens,
            value=cond_tokens,
        )

        return x + attn_out


class FinalLayerModulation(nn.Module):
    """Final layer with adaptive normalization.

    Used at the end of DiT to project tokens back to patch space.

    Parameters
    ----------
    hidden_dim : int
        Input hidden dimension.
    out_dim : int
        Output dimension (typically patch_size^2 * out_channels).
    cond_embed_dim : int
        Conditioning embedding dimension.
    """

    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        cond_embed_dim: int,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)

        # AdaLN produces shift and scale only (2 params)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_embed_dim, 2 * hidden_dim, bias=True),
        )

        self.proj = nn.Linear(hidden_dim, out_dim, bias=True)

        # Zero initialization
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        """Apply final layer transformation.

        Parameters
        ----------
        x : Tensor
            Input tokens of shape (B, N, hidden_dim).
        c : Tensor
            Conditioning embedding of shape (B, cond_embed_dim).

        Returns
        -------
        Tensor
            Output of shape (B, N, out_dim).
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        x = self.proj(x)
        return x
