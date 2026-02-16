"""Transformer blocks for DiT backbone.

This module provides factorized attention and transformer blocks
for processing spatio-temporal data.
"""

from typing import Literal, Optional

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from physicsflow.models.dit.modulation import (
    AdaLNModulation,
    CrossAttentionConditioning,
    FinalLayerModulation,
    modulate,
)


class Mlp(nn.Module):
    """MLP block with GELU activation.

    Parameters
    ----------
    in_features : int
        Input dimension.
    hidden_features : int, optional
        Hidden dimension. Defaults to in_features.
    out_features : int, optional
        Output dimension. Defaults to in_features.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Attention(nn.Module):
    """Multi-head self-attention using PyTorch's scaled_dot_product_attention.

    Uses Flash Attention when available for improved performance.

    Parameters
    ----------
    dim : int
        Input dimension.
    num_heads : int
        Number of attention heads.
    qkv_bias : bool
        Whether to use bias in QKV projection.
    attn_drop : float
        Attention dropout rate.
    proj_drop : float
        Output projection dropout rate.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})."
            )
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_drop_p = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        """Apply multi-head self-attention.

        Parameters
        ----------
        x : Tensor
            Input of shape (B, N, D).

        Returns
        -------
        Tensor
            Output of shape (B, N, D).
        """
        # Project to Q, K, V and reshape for multi-head attention
        qkv = self.qkv(x)
        qkv = rearrange(
            qkv, "b n (three h d) -> three b h n d",
            three=3, h=self.num_heads, d=self.head_dim
        )
        q, k, v = qkv.unbind(0)

        # Use PyTorch's scaled_dot_product_attention (uses Flash Attention when available)
        dropout_p = self.attn_drop_p if self.training else 0.0
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        # Reshape back to (B, N, D)
        x = rearrange(x, "b h n d -> b n (h d)")
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class FactorizedAttention(nn.Module):
    """Factorized spatial + temporal attention.

    Processes spatial and temporal dimensions separately for efficiency.
    Spatial attention operates on H'*W' patches per frame.
    Temporal attention operates on T frames per spatial position.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension.
    num_heads : int
        Number of attention heads.
    qkv_bias : bool
        Whether to use bias in QKV projection.
    attn_drop : float
        Attention dropout rate.
    proj_drop : float
        Output projection dropout rate.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()

        self.spatial_attn = Attention(
            dim=hidden_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

        self.temporal_attn = Attention(
            dim=hidden_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

    def forward(
        self,
        x: Tensor,
        num_frames: int,
        num_patches: int,
    ) -> Tensor:
        """Apply factorized spatial-temporal attention.

        Parameters
        ----------
        x : Tensor
            Input of shape (B, T*N, D) where N = num_spatial_patches.
        num_frames : int
            Number of temporal frames (T).
        num_patches : int
            Number of spatial patches (N = H'*W').

        Returns
        -------
        Tensor
            Output of shape (B, T*N, D).
        """
        # Spatial attention: (B, T*N, D) -> (B*T, N, D) -> attend -> (B, T*N, D)
        x = rearrange(x, "b (t n) d -> (b t) n d", t=num_frames, n=num_patches)
        x = self.spatial_attn(x)
        x = rearrange(x, "(b t) n d -> b (t n) d", t=num_frames)

        # Temporal attention: (B, T*N, D) -> (B*N, T, D) -> attend -> (B, T*N, D)
        x = rearrange(x, "b (t n) d -> (b n) t d", t=num_frames, n=num_patches)
        x = self.temporal_attn(x)
        x = rearrange(x, "(b n) t d -> b (t n) d", n=num_patches)

        return x


class DiTBlock(nn.Module):
    """Diffusion Transformer block with factorized attention.

    Supports both AdaLN-Zero and cross-attention conditioning.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension.
    num_heads : int
        Number of attention heads.
    mlp_ratio : float
        MLP hidden dimension ratio.
    cond_embed_dim : int
        Conditioning embedding dimension.
    conditioning_type : str
        'adaln' for AdaLN-Zero, 'cross_attention' for cross-attention.
    qkv_bias : bool
        Whether to use bias in QKV projection.
    dropout : float
        Dropout rate.
    attn_drop : float
        Attention dropout rate.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        cond_embed_dim: int = 256,
        conditioning_type: Literal["adaln", "cross_attention"] = "adaln",
        qkv_bias: bool = True,
        dropout: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conditioning_type = conditioning_type

        # Factorized attention
        self.attn = FactorizedAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=dropout,
        )

        # MLP
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=hidden_dim,
            hidden_features=mlp_hidden_dim,
            dropout=dropout,
        )

        # Conditioning
        if conditioning_type == "adaln":
            self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
            self.adaln = AdaLNModulation(hidden_dim, cond_embed_dim)
        else:
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.norm2 = nn.LayerNorm(hidden_dim)
            self.cross_attn = CrossAttentionConditioning(
                hidden_dim=hidden_dim,
                cond_embed_dim=cond_embed_dim,
                num_heads=num_heads,
                dropout=dropout,
            )

    def forward(
        self,
        x: Tensor,
        c: Tensor,
        num_frames: int,
        num_patches: int,
    ) -> Tensor:
        """Apply transformer block.

        Parameters
        ----------
        x : Tensor
            Input tokens of shape (B, T*N, D).
        c : Tensor
            Conditioning embedding of shape (B, cond_embed_dim).
        num_frames : int
            Number of temporal frames.
        num_patches : int
            Number of spatial patches.

        Returns
        -------
        Tensor
            Output tokens of shape (B, T*N, D).
        """
        if self.conditioning_type == "adaln":
            # AdaLN-Zero path
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln(c)

            # Attention with modulation
            x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
            x = x + gate_msa.unsqueeze(1) * self.attn(x_norm, num_frames, num_patches)

            # MLP with modulation
            x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
            x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        else:
            # Cross-attention path
            x = x + self.attn(self.norm1(x), num_frames, num_patches)
            x = self.cross_attn(x, c)
            x = x + self.mlp(self.norm2(x))

        return x


class FinalLayer(nn.Module):
    """Final layer to unpatchify tokens back to spatial output.

    Parameters
    ----------
    hidden_dim : int
        Input hidden dimension.
    out_channels : int
        Output channels.
    patch_size : tuple[int, int]
        Spatial patch size.
    cond_embed_dim : int
        Conditioning embedding dimension.
    """

    def __init__(
        self,
        hidden_dim: int,
        out_channels: int,
        patch_size: tuple[int, int],
        cond_embed_dim: int,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels

        out_dim = out_channels * patch_size[0] * patch_size[1]
        self.final = FinalLayerModulation(
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            cond_embed_dim=cond_embed_dim,
        )

    def forward(
        self,
        x: Tensor,
        c: Tensor,
        num_frames: int,
        num_patches_h: int,
        num_patches_w: int,
    ) -> Tensor:
        """Unpatchify tokens to spatial output.

        Parameters
        ----------
        x : Tensor
            Input tokens of shape (B, T*H'*W', hidden_dim).
        c : Tensor
            Conditioning embedding of shape (B, cond_embed_dim).
        num_frames : int
            Number of temporal frames.
        num_patches_h : int
            Number of patches in height.
        num_patches_w : int
            Number of patches in width.

        Returns
        -------
        Tensor
            Output of shape (B, C, T, H, W).
        """
        # Apply final modulation and projection
        x = self.final(x, c)  # (B, T*H'*W', C*ph*pw)

        # Unpatchify
        ph, pw = self.patch_size
        x = rearrange(
            x,
            "b (t h w) (c ph pw) -> b c t (h ph) (w pw)",
            t=num_frames,
            h=num_patches_h,
            w=num_patches_w,
            ph=ph,
            pw=pw,
            c=self.out_channels,
        )

        return x
