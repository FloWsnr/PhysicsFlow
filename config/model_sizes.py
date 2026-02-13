"""Configuration presets for DiT model sizes.

This module provides standard DiT configurations (S, B, L, XL) matching
the original DiT paper specifications.
"""

from dataclasses import dataclass


@dataclass
class DiTConfig:
    """Configuration for DiT model architecture.

    Parameters
    ----------
    hidden_dim : int
        Transformer hidden dimension.
    depth : int
        Number of transformer blocks.
    num_heads : int
        Number of attention heads.
    mlp_ratio : float
        MLP hidden dimension ratio (mlp_hidden = hidden_dim * mlp_ratio).
    patch_size : tuple[int, int]
        Spatial patch size.
    """

    hidden_dim: int
    depth: int
    num_heads: int
    mlp_ratio: float
    patch_size: tuple[int, int]
    time_embed_dim: int
    conditioning_type: str
    dropout: float
    attn_drop: float
    learnable_pos_embed: bool

    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.hidden_dim // self.num_heads

    def __post_init__(self):
        """Validate configuration."""
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )


# Standard DiT configurations from the original paper
# Approximate parameter counts assume 256x256 input with patch_size=2
DIT_CONFIGS: dict[str, DiTConfig] = {
    # DiT-S: ~33M parameters
    "DiT-S": DiTConfig(
        hidden_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        patch_size=(2, 2),
        time_embed_dim=256,
        conditioning_type="adaln",
        dropout=0.0,
        attn_drop=0.0,
        learnable_pos_embed=True,
    ),
    # DiT-B: ~130M parameters
    "DiT-B": DiTConfig(
        hidden_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        patch_size=(2, 2),
        time_embed_dim=256,
        conditioning_type="adaln",
        dropout=0.0,
        attn_drop=0.0,
        learnable_pos_embed=True,
    ),
    # DiT-L: ~458M parameters
    "DiT-L": DiTConfig(
        hidden_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        patch_size=(2, 2),
        time_embed_dim=256,
        conditioning_type="adaln",
        dropout=0.0,
        attn_drop=0.0,
        learnable_pos_embed=True,
    ),
    # DiT-XL: ~675M parameters
    "DiT-XL": DiTConfig(
        hidden_dim=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        patch_size=(2, 2),
        time_embed_dim=256,
        conditioning_type="adaln",
        dropout=0.0,
        attn_drop=0.0,
        learnable_pos_embed=True,
    ),
}


def get_dit_config(model_size: str) -> DiTConfig:
    """Get DiT configuration by size name.

    Parameters
    ----------
    model_size : str
        Model size name: 'DiT-S', 'DiT-B', 'DiT-L', or 'DiT-XL'.

    Returns
    -------
    DiTConfig
        Configuration for the specified model size.

    Raises
    ------
    ValueError
        If model size is not recognized.
    """
    if model_size not in DIT_CONFIGS:
        raise ValueError(
            f"Unknown DiT size: {model_size}. "
            f"Available sizes: {list(DIT_CONFIGS.keys())}"
        )
    return DIT_CONFIGS[model_size]


def list_dit_configs() -> list[str]:
    """List available DiT configuration names.

    Returns
    -------
    list[str]
        List of available configuration names.
    """
    return list(DIT_CONFIGS.keys())
