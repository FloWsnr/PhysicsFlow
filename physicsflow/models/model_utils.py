"""Model factory utilities."""

import torch.nn as nn


def get_model(model_config: dict) -> nn.Module:
    """Factory function to create models from config.

    Parameters
    ----------
    model_config : dict
        Configuration dictionary with 'type' key specifying model type.
        Required keys depend on model type:

        For 'flow_matching':
            - in_channels: int
            - spatial_size: list[int, int]
            - temporal_size: int
            - cond_dim: int (number of physics parameters)
            - backbone: str ('mlp' or 'dit', default 'mlp')
            - scheduler: str (optional, default 'cond_ot')
            - scheduler_kwargs: dict (optional)

            For backbone='mlp':
                - hidden_dim: int (optional, default 256)

            For backbone='dit':
                - dit_size: str ('DiT-S', 'DiT-B', 'DiT-L', 'DiT-XL')
                - hidden_dim: int (optional, overrides dit_size)
                - depth: int (optional, overrides dit_size)
                - num_heads: int (optional, overrides dit_size)
                - mlp_ratio: float (optional, default 4.0)
                - patch_size: list[int, int] (optional, default [2, 2])
                - conditioning_type: str ('adaln' or 'cross_attention')
                - dropout: float (optional, default 0.0)
                - gradient_checkpointing: bool (optional, default False)

    Returns
    -------
    nn.Module
        The created model.

    Raises
    ------
    ValueError
        If model type is not recognized.
    """
    model_type = model_config.get("type", "placeholder")

    if model_type == "flow_matching":
        return _create_flow_matching_model(model_config)
    elif model_type == "placeholder":
        return nn.Module()
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available: 'flow_matching', 'placeholder'"
        )


def _create_flow_matching_model(config: dict) -> nn.Module:
    """Create a FlowMatchingModel from config."""
    from physicsflow.models.flow_matching import FlowMatchingModel
    from physicsflow.models.backbone import MLPBackbone, DiTBackbone, get_dit_config

    # Required parameters
    in_channels = config["in_channels"]
    spatial_size = tuple(config["spatial_size"])
    temporal_size = config["temporal_size"]
    cond_dim = config.get("cond_dim", 0)

    # Backbone type
    backbone_type = config.get("backbone", "mlp")

    # Scheduler parameters
    scheduler = config.get("scheduler", "cond_ot")
    scheduler_kwargs = config.get("scheduler_kwargs", {})

    if backbone_type == "mlp":
        # MLP backbone
        velocity_net = MLPBackbone(
            in_channels=in_channels,
            spatial_size=spatial_size,
            temporal_size=temporal_size,
            cond_dim=cond_dim,
            hidden_dim=config.get("hidden_dim", 256),
            time_embed_dim=config.get("time_embed_dim", 64),
        )
    elif backbone_type == "dit":
        # DiT backbone
        dit_size = config.get("dit_size", "DiT-S")
        dit_config = get_dit_config(dit_size)

        velocity_net = DiTBackbone(
            in_channels=in_channels,
            spatial_size=spatial_size,
            temporal_size=temporal_size,
            cond_dim=cond_dim,
            hidden_dim=config.get("hidden_dim", dit_config.hidden_dim),
            depth=config.get("depth", dit_config.depth),
            num_heads=config.get("num_heads", dit_config.num_heads),
            mlp_ratio=config.get("mlp_ratio", dit_config.mlp_ratio),
            patch_size=tuple(config.get("patch_size", list(dit_config.patch_size))),
            time_embed_dim=config.get("time_embed_dim", 256),
            conditioning_type=config.get("conditioning_type", "adaln"),
            dropout=config.get("dropout", 0.0),
            attn_drop=config.get("attn_drop", 0.0),
            learnable_pos_embed=config.get("learnable_pos_embed", True),
            use_gradient_checkpointing=config.get("gradient_checkpointing", False),
        )
    else:
        raise ValueError(
            f"Unknown backbone type: {backbone_type}. "
            f"Available: 'mlp', 'dit'"
        )

    return FlowMatchingModel(
        velocity_net=velocity_net,
        scheduler=scheduler,
        scheduler_kwargs=scheduler_kwargs,
    )
