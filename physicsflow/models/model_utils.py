"""Model factory utilities."""

import torch.nn as nn

from physicsflow.models.flow_matching import FlowMatchingModel, get_scheduler
from physicsflow.models.dit import DiTBackbone


def get_model(model_config: dict) -> nn.Module:
    """Factory function to create models from config.

    Parameters
    ----------
    model_config : dict
        Configuration dictionary

    Returns
    -------
    nn.Module
        The created flow matching model.
    """
    return _create_flow_matching_model(model_config)


def _create_flow_matching_model(config: dict) -> nn.Module:
    """Create a FlowMatchingModel from config."""

    velocity_net = DiTBackbone(
        in_channels=config["in_channels"],
        spatial_size=tuple(config["spatial_size"]),
        temporal_size=config["temporal_size"],
        cond_dim=config["cond_dim"],
        hidden_dim=config["hidden_dim"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        mlp_ratio=config["mlp_ratio"],
        patch_size=tuple(config["patch_size"]),
        time_embed_dim=config["time_embed_dim"],
        conditioning_type=config["conditioning_type"],
        dropout=config["dropout"],
        attn_drop=config["attn_drop"],
        learnable_pos_embed=config["learnable_pos_embed"],
        use_gradient_checkpointing=config["gradient_checkpointing"],
    )

    scheduler = get_scheduler(
        config["scheduler"]["type"], **config["scheduler"]["params"]
    )

    return FlowMatchingModel(
        velocity_net=velocity_net,
        scheduler=scheduler,
    )
