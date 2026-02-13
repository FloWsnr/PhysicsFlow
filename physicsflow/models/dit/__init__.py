"""DiT (Diffusion Transformer) backbone for flow matching.

This module provides a transformer-based backbone for velocity prediction
in flow matching models, optimized for spatio-temporal physics data.
"""

from physicsflow.models.backbone.dit.blocks import (
    Attention,
    DiTBlock,
    FactorizedAttention,
    FinalLayer,
    Mlp,
)
from physicsflow.models.backbone.dit.configs import (
    DIT_CONFIGS,
    DiTConfig,
    get_dit_config,
    list_dit_configs,
)
from physicsflow.models.backbone.dit.dit_backbone import DiTBackbone
from physicsflow.models.backbone.dit.embeddings import (
    PatchEmbed3D,
    SpatioTemporalPosEmbed,
)
from physicsflow.models.backbone.dit.modulation import (
    AdaLNBlock,
    AdaLNModulation,
    CrossAttentionConditioning,
    FinalLayerModulation,
    modulate,
)

__all__ = [
    # Main backbone
    "DiTBackbone",
    # Configurations
    "DiTConfig",
    "DIT_CONFIGS",
    "get_dit_config",
    "list_dit_configs",
    # Embeddings
    "PatchEmbed3D",
    "SpatioTemporalPosEmbed",
    # Blocks
    "DiTBlock",
    "FactorizedAttention",
    "FinalLayer",
    "Attention",
    "Mlp",
    # Modulation
    "AdaLNModulation",
    "AdaLNBlock",
    "CrossAttentionConditioning",
    "FinalLayerModulation",
    "modulate",
]
