"""Embedding layers for DiT backbone.

This module provides patch embedding and positional embedding layers
for processing 5D physics simulation data (B, C, T, H, W).
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor


class PatchEmbed3D(nn.Module):
    """Convert 5D tensor to patch tokens.

    Patchifies spatial dimensions while keeping temporal dimension intact.
    Input: (B, C, T, H, W) -> Output: (B, T*H'*W', hidden_dim)
    where H' = H/patch_h, W' = W/patch_w.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    hidden_dim : int
        Output embedding dimension.
    patch_size : tuple[int, int]
        Spatial patch size (patch_h, patch_w).
    bias : bool
        Whether to use bias in projection.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        patch_size: tuple[int, int] = (2, 2),
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size

        # Conv3d with kernel (1, ph, pw) to keep temporal dim intact
        self.proj = nn.Conv3d(
            in_channels,
            hidden_dim,
            kernel_size=(1, patch_size[0], patch_size[1]),
            stride=(1, patch_size[0], patch_size[1]),
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Patchify input tensor.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (B, C, T, H, W).

        Returns
        -------
        Tensor
            Patch tokens of shape (B, T*H'*W', hidden_dim).
        """
        # (B, C, T, H, W) -> (B, hidden_dim, T, H', W')
        x = self.proj(x)
        # (B, hidden_dim, T, H', W') -> (B, T*H'*W', hidden_dim)
        x = rearrange(x, "b d t h w -> b (t h w) d")
        return x

    def get_num_patches(
        self, spatial_size: tuple[int, int], temporal_size: int
    ) -> tuple[int, int, int]:
        """Get number of patches for given input size.

        Parameters
        ----------
        spatial_size : tuple[int, int]
            Input spatial size (H, W).
        temporal_size : int
            Number of temporal frames.

        Returns
        -------
        tuple[int, int, int]
            (num_patches_h, num_patches_w, num_frames)
        """
        num_patches_h = spatial_size[0] // self.patch_size[0]
        num_patches_w = spatial_size[1] // self.patch_size[1]
        return num_patches_h, num_patches_w, temporal_size


class SpatioTemporalPosEmbed(nn.Module):
    """Factorized spatial + temporal positional embeddings.

    Uses separate embeddings for spatial and temporal dimensions,
    which are added together to form the final positional encoding.

    Parameters
    ----------
    hidden_dim : int
        Embedding dimension.
    max_spatial_size : tuple[int, int]
        Maximum (H', W') patch grid size.
    max_temporal_size : int
        Maximum number of temporal frames.
    learnable : bool
        If True, use learnable embeddings. Otherwise, use sinusoidal.
    """

    def __init__(
        self,
        hidden_dim: int,
        max_spatial_size: tuple[int, int] = (64, 64),
        max_temporal_size: int = 64,
        learnable: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_spatial_size = max_spatial_size
        self.max_temporal_size = max_temporal_size
        self.learnable = learnable

        max_spatial_patches = max_spatial_size[0] * max_spatial_size[1]

        if learnable:
            self.spatial_embed = nn.Parameter(
                torch.zeros(1, max_spatial_patches, hidden_dim)
            )
            self.temporal_embed = nn.Parameter(
                torch.zeros(1, max_temporal_size, hidden_dim)
            )
            # Initialize with truncated normal
            nn.init.trunc_normal_(self.spatial_embed, std=0.02)
            nn.init.trunc_normal_(self.temporal_embed, std=0.02)
        else:
            # Register sinusoidal embeddings as buffers
            spatial_embed = self._create_sinusoidal_embedding(
                max_spatial_patches, hidden_dim
            )
            temporal_embed = self._create_sinusoidal_embedding(
                max_temporal_size, hidden_dim
            )
            self.register_buffer("spatial_embed", spatial_embed.unsqueeze(0))
            self.register_buffer("temporal_embed", temporal_embed.unsqueeze(0))

    def _create_sinusoidal_embedding(self, num_positions: int, dim: int) -> Tensor:
        """Create sinusoidal positional embeddings.

        Parameters
        ----------
        num_positions : int
            Number of positions.
        dim : int
            Embedding dimension.

        Returns
        -------
        Tensor
            Positional embeddings of shape (num_positions, dim).
        """
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        pos = torch.arange(num_positions, dtype=torch.float32)
        emb = pos[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_positions, 1)], dim=1)
        return emb

    def forward(
        self,
        num_patches_h: int,
        num_patches_w: int,
        num_frames: int,
    ) -> Tensor:
        """Compute positional embeddings for given dimensions.

        Parameters
        ----------
        num_patches_h : int
            Number of patches in height dimension.
        num_patches_w : int
            Number of patches in width dimension.
        num_frames : int
            Number of temporal frames.

        Returns
        -------
        Tensor
            Positional embeddings of shape (1, T*H'*W', hidden_dim).
        """
        num_spatial = num_patches_h * num_patches_w

        # Get spatial embeddings: (1, H'*W', D)
        spatial = self.spatial_embed[:, :num_spatial, :]

        # Get temporal embeddings: (1, T, D)
        temporal = self.temporal_embed[:, :num_frames, :]

        # Combine via broadcasting:
        # spatial: (1, 1, H'*W', D) + temporal: (1, T, 1, D) -> (1, T, H'*W', D)
        spatial = spatial.unsqueeze(1)  # (1, 1, H'*W', D)
        temporal = temporal.unsqueeze(2)  # (1, T, 1, D)

        pos_embed = spatial + temporal  # (1, T, H'*W', D)
        pos_embed = rearrange(pos_embed, "b t s d -> b (t s) d")

        return pos_embed

    def interpolate(
        self,
        num_patches_h: int,
        num_patches_w: int,
        num_frames: int,
    ) -> Tensor:
        """Interpolate positional embeddings for different input sizes.

        Useful when fine-tuning on different resolutions than training.

        Parameters
        ----------
        num_patches_h : int
            Target number of patches in height.
        num_patches_w : int
            Target number of patches in width.
        num_frames : int
            Target number of frames.

        Returns
        -------
        Tensor
            Interpolated positional embeddings.
        """
        # Spatial interpolation
        spatial = self.spatial_embed  # (1, max_h*max_w, D)
        max_h, max_w = self.max_spatial_size
        spatial = rearrange(spatial, "b (h w) d -> b d h w", h=max_h, w=max_w)
        spatial = nn.functional.interpolate(
            spatial,
            size=(num_patches_h, num_patches_w),
            mode="bilinear",
            align_corners=False,
        )
        spatial = rearrange(spatial, "b d h w -> b (h w) d")

        # Temporal interpolation
        temporal = self.temporal_embed  # (1, max_t, D)
        temporal = rearrange(temporal, "b t d -> b d t")
        temporal = nn.functional.interpolate(
            temporal, size=num_frames, mode="linear", align_corners=False
        )
        temporal = rearrange(temporal, "b d t -> b t d")

        # Combine
        spatial = spatial.unsqueeze(1)
        temporal = temporal.unsqueeze(2)
        pos_embed = spatial + temporal
        pos_embed = rearrange(pos_embed, "b t s d -> b (t s) d")

        return pos_embed
