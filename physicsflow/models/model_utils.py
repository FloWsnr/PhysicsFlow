"""Model factory utilities."""

import torch.nn as nn

from physicsflow.models.flow_matching import FlowMatchingModel, get_scheduler
from physicsflow.models.dit import DiTBackbone
from config.model_sizes import get_dit_config


def get_model(model_config: dict) -> nn.Module:
    """Factory function to create models from config.

    Parameters
    ----------
    model_config : dict
        Configuration dictionary. Must contain a ``size`` key (e.g. "S", "B",
        "L", "XL") that selects architecture hyper-parameters from the
        predefined DiT configurations.

    Returns
    -------
    nn.Module
        The created flow matching model.
    """
    return _create_flow_matching_model(model_config)


def _create_flow_matching_model(config: dict) -> nn.Module:
    """Create a FlowMatchingModel from config."""

    dit_config = get_dit_config(f"DiT-{config['size']}")

    velocity_net = DiTBackbone(
        in_channels=config["in_channels"],
        spatial_size=tuple(config["spatial_size"]),
        temporal_size=config["temporal_size"],
        cond_dim=config["cond_dim"],
        hidden_dim=dit_config.hidden_dim,
        depth=dit_config.depth,
        num_heads=dit_config.num_heads,
        mlp_ratio=dit_config.mlp_ratio,
        patch_size=dit_config.patch_size,
        time_embed_dim=dit_config.time_embed_dim,
        conditioning_type=dit_config.conditioning_type,
        dropout=dit_config.dropout,
        attn_drop=dit_config.attn_drop,
        learnable_pos_embed=dit_config.learnable_pos_embed,
    )

    scheduler = get_scheduler(
        config["scheduler"]["type"], **config["scheduler"]["params"]
    )

    return FlowMatchingModel(
        velocity_net=velocity_net,
        scheduler=scheduler,
    )
