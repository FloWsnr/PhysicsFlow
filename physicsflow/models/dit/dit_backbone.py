"""Diffusion Transformer (DiT) backbone for flow matching.

This module provides the main DiT backbone class that can be used
as a velocity network in flow matching models.
"""

from typing import Literal, Optional

import torch
import torch.nn as nn
from torch import Tensor

from physicsflow.models.dit.blocks import DiTBlock, FinalLayer
from config.model_sizes import DiTConfig, get_dit_config
from physicsflow.models.dit.embeddings import PatchEmbed3D, SpatioTemporalPosEmbed
from physicsflow.models.common import ConditioningProjection, TimeEmbedding


class DiTBackbone(nn.Module):
    """Diffusion Transformer backbone for flow matching.

    Matches the interface of MLPBackbone for drop-in replacement.
    Implements factorized spatial-temporal attention for 5D physics data.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    spatial_size : tuple[int, int]
        Spatial dimensions (H, W).
    temporal_size : int
        Number of time steps in data.
    cond_dim : int
        Dimension of conditioning vector (physics parameters).
        Set to 0 for unconditional generation.
    hidden_dim : int
        Transformer hidden dimension.
    depth : int
        Number of transformer blocks.
    num_heads : int
        Number of attention heads.
    mlp_ratio : float
        MLP hidden dimension ratio.
    patch_size : tuple[int, int]
        Spatial patch size.
    time_embed_dim : int
        Time embedding dimension.
    conditioning_type : str
        'adaln' for AdaLN-Zero, 'cross_attention' for cross-attention.
    dropout : float
        Dropout rate.
    attn_drop : float
        Attention dropout rate.
    learnable_pos_embed : bool
        Whether to use learnable positional embeddings.

    Examples
    --------
    >>> dit = DiTBackbone(
    ...     in_channels=3,
    ...     spatial_size=(64, 64),
    ...     temporal_size=10,
    ...     cond_dim=5,
    ...     hidden_dim=384,
    ...     depth=12,
    ...     num_heads=6,
    ... )
    >>> x_t = torch.randn(2, 3, 10, 64, 64)
    >>> t = torch.rand(2)
    >>> cond = torch.randn(2, 5)
    >>> v = dit(x_t, t, cond)
    >>> v.shape
    torch.Size([2, 3, 10, 64, 64])
    """

    def __init__(
        self,
        in_channels: int,
        spatial_size: tuple[int, int],
        temporal_size: int,
        cond_dim: int,
        hidden_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        patch_size: tuple[int, int],
        time_embed_dim: int,
        conditioning_type: Literal["adaln", "cross_attention"],
        dropout: float,
        attn_drop: float,
        learnable_pos_embed: bool,
    ):
        super().__init__()

        # Validate inputs
        if spatial_size[0] % patch_size[0] != 0 or spatial_size[1] % patch_size[1] != 0:
            raise ValueError(
                f"Spatial size {spatial_size} must be divisible by patch size {patch_size}"
            )
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )

        # Store configuration
        self.in_channels = in_channels
        self.spatial_size = spatial_size
        self.temporal_size = temporal_size
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.time_embed_dim = time_embed_dim

        # Compute derived dimensions
        self.num_patches_h = spatial_size[0] // patch_size[0]
        self.num_patches_w = spatial_size[1] // patch_size[1]
        self.num_spatial_patches = self.num_patches_h * self.num_patches_w

        # Combined conditioning embedding dimension
        cond_embed_dim = time_embed_dim

        # Patch embedding
        self.patch_embed = PatchEmbed3D(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            patch_size=patch_size,
        )

        # Positional embedding
        self.pos_embed = SpatioTemporalPosEmbed(
            hidden_dim=hidden_dim,
            max_spatial_size=(self.num_patches_h, self.num_patches_w),
            max_temporal_size=temporal_size,
            learnable=learnable_pos_embed,
        )

        # Time embedding (reuse existing)
        self.time_embed = TimeEmbedding(dim=time_embed_dim)

        # Conditioning projection (reuse existing if cond_dim > 0)
        if cond_dim > 0:
            self.cond_embed = ConditioningProjection(
                input_dim=cond_dim,
                output_dim=time_embed_dim,
            )
        else:
            self.cond_embed = None

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                cond_embed_dim=cond_embed_dim,
                conditioning_type=conditioning_type,
                dropout=dropout,
                attn_drop=attn_drop,
            )
            for _ in range(depth)
        ])

        # Final layer
        self.final_layer = FinalLayer(
            hidden_dim=hidden_dim,
            out_channels=in_channels,
            patch_size=patch_size,
            cond_embed_dim=cond_embed_dim,
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with truncated normal."""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_basic_init)

    @classmethod
    def from_config(
        cls,
        config: DiTConfig | str,
        in_channels: int,
        spatial_size: tuple[int, int],
        temporal_size: int,
        cond_dim: int = 0,
        **kwargs,
    ) -> "DiTBackbone":
        """Create DiTBackbone from a configuration.

        Parameters
        ----------
        config : DiTConfig or str
            DiTConfig instance or size name ('DiT-S', 'DiT-B', etc.).
        in_channels : int
            Number of input channels.
        spatial_size : tuple[int, int]
            Spatial dimensions (H, W).
        temporal_size : int
            Number of time steps.
        cond_dim : int
            Conditioning dimension.
        **kwargs
            Additional arguments to override config values.

        Returns
        -------
        DiTBackbone
            Configured DiT backbone.
        """
        if isinstance(config, str):
            config = get_dit_config(config)

        config_fields = [
            "hidden_dim", "depth", "num_heads", "mlp_ratio", "patch_size",
            "time_embed_dim", "conditioning_type", "dropout", "attn_drop",
            "learnable_pos_embed",
        ]
        config_kwargs = {
            field: kwargs.get(field, getattr(config, field))
            for field in config_fields
        }
        extra_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in config_fields
        }

        return cls(
            in_channels=in_channels,
            spatial_size=spatial_size,
            temporal_size=temporal_size,
            cond_dim=cond_dim,
            **config_kwargs,
            **extra_kwargs,
        )

    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        cond: Optional[Tensor] = None,
    ) -> Tensor:
        """Predict velocity for flow matching.

        Parameters
        ----------
        x_t : Tensor
            Interpolated samples, shape (B, C, T, H, W).
        t : Tensor
            Time values in [0, 1], shape (B,).
        cond : Tensor, optional
            Physics parameters, shape (B, cond_dim).

        Returns
        -------
        Tensor
            Predicted velocity, shape (B, C, T, H, W).
        """
        B, C, T, H, W = x_t.shape

        # Validate input dimensions
        assert C == self.in_channels, f"Expected {self.in_channels} channels, got {C}"
        assert T == self.temporal_size, f"Expected {self.temporal_size} frames, got {T}"
        assert (H, W) == self.spatial_size, f"Expected {self.spatial_size} spatial, got ({H}, {W})"

        # Embed patches: (B, C, T, H, W) -> (B, T*H'*W', hidden_dim)
        x = self.patch_embed(x_t)

        # Add positional embedding
        pos = self.pos_embed(self.num_patches_h, self.num_patches_w, T)
        x = x + pos

        # Embed timestep: (B,) -> (B, time_embed_dim)
        c = self.time_embed(t)

        # Add conditioning if present
        if cond is not None and self.cond_embed is not None:
            c = c + self.cond_embed(cond)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, c, T, self.num_spatial_patches)

        # Final layer: unpatchify
        v = self.final_layer(x, c, T, self.num_patches_h, self.num_patches_w)

        return v

    def get_num_params(self, non_embedding: bool = False) -> int:
        """Get number of parameters.

        Parameters
        ----------
        non_embedding : bool
            If True, exclude embedding parameters.

        Returns
        -------
        int
            Number of parameters.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.pos_embed.spatial_embed.numel()
            n_params -= self.pos_embed.temporal_embed.numel()
        return n_params

    def extra_repr(self) -> str:
        """Extra representation string."""
        return (
            f"in_channels={self.in_channels}, "
            f"spatial_size={self.spatial_size}, "
            f"temporal_size={self.temporal_size}, "
            f"hidden_dim={self.hidden_dim}, "
            f"patch_size={self.patch_size}, "
            f"depth={len(self.blocks)}, "
            f"params={self.get_num_params():,}"
        )
