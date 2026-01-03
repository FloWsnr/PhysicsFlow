"""Model factory utilities."""

import torch
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
            - hidden_dim: int (optional, default 256)
            - scheduler: str (optional, default 'cond_ot')
            - scheduler_kwargs: dict (optional)

        For 'diffusion':
            - in_channels: int
            - spatial_size: list[int, int]
            - temporal_size: int
            - cond_dim: int (number of physics parameters)
            - hidden_dim: int (optional, default 256)
            - timesteps: int (optional, default 1000)
            - schedule: str (optional, default 'cosine')
            - objective: str (optional, default 'pred_noise')
            - loss_type: str (optional, default 'l2')
            - snr_weighting: bool (optional, default False)

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
    elif model_type == "diffusion":
        return _create_diffusion_model(model_config)
    elif model_type == "placeholder":
        return nn.Module()
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available: 'flow_matching', 'diffusion', 'placeholder'"
        )


def _create_flow_matching_model(config: dict) -> nn.Module:
    """Create a FlowMatchingModel from config."""
    from physicsflow.models.flow_matching import (
        FlowMatchingModel,
        PlaceholderVelocityNet,
    )

    # Required parameters
    in_channels = config["in_channels"]
    spatial_size = tuple(config["spatial_size"])
    temporal_size = config["temporal_size"]
    cond_dim = config.get("cond_dim", 0)

    # Optional parameters
    hidden_dim = config.get("hidden_dim", 256)
    scheduler = config.get("scheduler", "cond_ot")
    scheduler_kwargs = config.get("scheduler_kwargs", {})

    # Create velocity network (placeholder for now)
    velocity_net = PlaceholderVelocityNet(
        in_channels=in_channels,
        spatial_size=spatial_size,
        temporal_size=temporal_size,
        cond_dim=cond_dim,
        hidden_dim=hidden_dim,
    )

    return FlowMatchingModel(
        velocity_net=velocity_net,
        scheduler=scheduler,
        scheduler_kwargs=scheduler_kwargs,
    )


def _create_diffusion_model(config: dict) -> nn.Module:
    """Create a DiffusionModel from config."""
    from physicsflow.models.diffusion import (
        DiffusionModel,
        PlaceholderDenoiser,
    )

    # Required parameters
    in_channels = config["in_channels"]
    spatial_size = tuple(config["spatial_size"])
    temporal_size = config["temporal_size"]
    cond_dim = config.get("cond_dim", 0)

    # Optional parameters
    hidden_dim = config.get("hidden_dim", 256)
    timesteps = config.get("timesteps", 1000)
    schedule = config.get("schedule", "cosine")
    objective = config.get("objective", "pred_noise")
    loss_type = config.get("loss_type", "l2")
    snr_weighting = config.get("snr_weighting", False)
    min_snr_gamma = config.get("min_snr_gamma", 5.0)

    # Create denoiser network (placeholder for now)
    network = PlaceholderDenoiser(
        in_channels=in_channels,
        spatial_size=spatial_size,
        temporal_size=temporal_size,
        cond_dim=cond_dim,
        hidden_dim=hidden_dim,
    )

    return DiffusionModel(
        network=network,
        timesteps=timesteps,
        schedule=schedule,
        objective=objective,
        loss_type=loss_type,
        snr_weighting=snr_weighting,
        min_snr_gamma=min_snr_gamma,
    )
