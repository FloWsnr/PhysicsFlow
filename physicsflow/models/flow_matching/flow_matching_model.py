"""Flow Matching model implementation.

Flow Matching learns a velocity field that transports samples from
a simple prior distribution (Gaussian noise) to the data distribution.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from physicsflow.models.common import sinusoidal_embedding
from physicsflow.models.flow_matching.schedulers import BaseScheduler, get_scheduler


@dataclass
class FlowMatchingOutput:
    """Output from FlowMatchingModel.forward().

    Attributes
    ----------
    loss : Tensor
        Training loss (MSE between predicted and target velocity).
    predicted_velocity : Tensor
        Model's velocity prediction.
    target_velocity : Tensor
        Ground truth velocity.
    x_t : Tensor
        Noised samples at time t.
    """

    loss: Tensor
    predicted_velocity: Tensor
    target_velocity: Tensor
    x_t: Tensor

    # Aliases for compatibility with trainer metrics
    @property
    def pred(self) -> Tensor:
        return self.predicted_velocity

    @property
    def target(self) -> Tensor:
        return self.target_velocity


class FlowMatchingModel(nn.Module):
    """Flow Matching model wrapper for physics simulations.

    This class wraps a velocity prediction network and handles:
    - Noise sampling and path interpolation
    - Velocity target computation
    - Loss calculation during training
    - ODE integration for sampling

    The model learns to predict the velocity field v_theta(x_t, t, c) where:
    - x_t is the noised sample at time t
    - t is the time step in [0, 1]
    - c is the conditioning (physics parameters)

    Parameters
    ----------
    velocity_net : nn.Module
        Neural network that predicts velocity. Should accept:
        - x_t: (B, C, T, H, W) noised samples
        - t: (B,) time steps
        - cond: (B, num_params) physics parameters (optional)
        And return predicted velocity of same shape as x_t.
    scheduler : BaseScheduler or str
        Flow matching scheduler instance or name ('cond_ot', 'cosine', etc.).
    scheduler_kwargs : dict, optional
        Additional arguments if scheduler is a string.
    """

    def __init__(
        self,
        velocity_net: nn.Module,
        scheduler: BaseScheduler | str = "cond_ot",
        scheduler_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.velocity_net = velocity_net

        if isinstance(scheduler, str):
            scheduler_kwargs = scheduler_kwargs or {}
            self.scheduler = get_scheduler(scheduler, **scheduler_kwargs)
        else:
            self.scheduler = scheduler

    def forward(self, data: dict[str, Tensor]) -> FlowMatchingOutput:
        """Forward pass for training.

        Computes the flow matching loss:
            L = E_{t, x_0, x_1}[||v_theta(x_t, t, c) - v_target||^2]

        Parameters
        ----------
        data : dict
            Dictionary containing:
            - 'input_fields': Target data (x_1), shape (B, C, T, H, W)
            - 'constant_scalars': Physics parameters, shape (B, num_params)

        Returns
        -------
        FlowMatchingOutput
            Dataclass with loss, predicted_velocity, target_velocity, x_t.
        """
        x_1 = data["input_fields"]
        cond = data.get("constant_scalars", None)

        batch_size = x_1.shape[0]
        device = x_1.device
        dtype = x_1.dtype

        # Sample noise
        x_0 = torch.randn_like(x_1)

        # Sample time uniformly in [0, 1]
        t = torch.rand(batch_size, device=device, dtype=dtype)

        # Compute interpolated samples and target velocity
        x_t = self.scheduler.sample_path(x_0, x_1, t)
        v_target = self.scheduler.target_velocity(x_0, x_1, t)

        # Predict velocity
        v_pred = self.velocity_net(x_t, t, cond)

        # Compute MSE loss
        loss = F.mse_loss(v_pred, v_target)

        return FlowMatchingOutput(
            loss=loss,
            predicted_velocity=v_pred,
            target_velocity=v_target,
            x_t=x_t,
        )

    @torch.inference_mode()
    def sample(
        self,
        shape: tuple[int, ...],
        cond: Optional[Tensor] = None,
        num_steps: int = 100,
        method: str = "euler",
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """Generate samples by integrating the learned velocity field.

        Solves the ODE: dx/dt = v_theta(x, t, c) from t=0 to t=1.

        Parameters
        ----------
        shape : tuple
            Shape of samples to generate (B, C, T, H, W).
        cond : Tensor, optional
            Physics parameters, shape (B, num_params).
        num_steps : int
            Number of integration steps.
        method : str
            Integration method: 'euler' or 'midpoint'.
        device : torch.device, optional
            Device to generate samples on.

        Returns
        -------
        Tensor
            Generated samples at t=1, shape (B, C, T, H, W).
        """
        if device is None:
            device = next(self.parameters()).device

        batch_size = shape[0]
        dt = 1.0 / num_steps

        # Start from noise
        x = torch.randn(shape, device=device)

        for step in range(num_steps):
            t = torch.full((batch_size,), step * dt, device=device)

            if method == "euler":
                v = self.velocity_net(x, t, cond)
                x = x + v * dt
            elif method == "midpoint":
                # Midpoint method (RK2)
                v1 = self.velocity_net(x, t, cond)
                x_mid = x + v1 * (dt / 2)
                t_mid = t + dt / 2
                v_mid = self.velocity_net(x_mid, t_mid, cond)
                x = x + v_mid * dt
            else:
                raise ValueError(f"Unknown integration method: {method}")

        return x


class PlaceholderVelocityNet(nn.Module):
    """Placeholder 1-layer MLP velocity network for testing.

    This is a simple implementation to verify the flow matching
    infrastructure works correctly. Replace with a proper architecture
    (e.g., U-Net, Transformer) for actual training.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    spatial_size : tuple
        Spatial dimensions (H, W).
    temporal_size : int
        Number of time steps in data.
    cond_dim : int
        Dimension of conditioning vector (physics parameters).
    hidden_dim : int
        Hidden layer dimension.
    time_embed_dim : int
        Dimension of time embedding.
    """

    def __init__(
        self,
        in_channels: int,
        spatial_size: tuple[int, int],
        temporal_size: int,
        cond_dim: int = 0,
        hidden_dim: int = 256,
        time_embed_dim: int = 64,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.spatial_size = spatial_size
        self.temporal_size = temporal_size
        self.cond_dim = cond_dim
        self.time_embed_dim = time_embed_dim

        # Compute flattened size
        self.flat_size = in_channels * temporal_size * spatial_size[0] * spatial_size[1]

        # Input: flattened x_t + time embedding + conditioning
        input_dim = self.flat_size + time_embed_dim + cond_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.flat_size),
        )

    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        cond: Optional[Tensor] = None,
    ) -> Tensor:
        """Predict velocity.

        Parameters
        ----------
        x_t : Tensor
            Noised samples, shape (B, C, T, H, W).
        t : Tensor
            Time steps in [0, 1], shape (B,).
        cond : Tensor, optional
            Physics parameters, shape (B, cond_dim).

        Returns
        -------
        Tensor
            Predicted velocity, shape (B, C, T, H, W).
        """
        batch_size = x_t.shape[0]
        original_shape = x_t.shape

        # Flatten spatial dimensions
        x_flat = x_t.reshape(batch_size, -1)

        # Sinusoidal time embedding
        t_emb = sinusoidal_embedding(t, self.time_embed_dim)

        # Concatenate inputs
        if cond is not None:
            inputs = torch.cat([x_flat, t_emb, cond], dim=-1)
        else:
            inputs = torch.cat([x_flat, t_emb], dim=-1)

        # Forward through network
        output = self.net(inputs)

        # Reshape to original spatial dimensions
        return output.reshape(*original_shape)
