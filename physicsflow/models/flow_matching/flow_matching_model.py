"""Flow Matching model implementation.

Flow Matching learns a velocity field that transports samples from
a simple prior distribution (Gaussian noise) to the data distribution.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from physicsflow.models.flow_matching.schedulers import BaseScheduler


@dataclass
class FlowMatchingOutput:
    """Output from FlowMatchingModel.forward().

    Attributes
    ----------
    predicted_velocity : Tensor
        Model's velocity prediction.
    target_velocity : Tensor
        Ground truth velocity.
    x_t : Tensor
        Noised samples at time t.
    """

    predicted_velocity: Tensor
    target_velocity: Tensor
    x_t: Tensor


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
    scheduler : BaseScheduler
        Flow matching scheduler instance
    """

    def __init__(
        self,
        velocity_net: nn.Module,
        scheduler: BaseScheduler,
    ):
        super().__init__()
        self.velocity_net = velocity_net
        self.scheduler = scheduler

    def forward(
        self, x_1: torch.Tensor, cond: Optional[torch.Tensor] = None
    ) -> FlowMatchingOutput:
        """Forward pass for training.

        Computes the predicted velocity:
            v_theta(x_t, t, c)

        Parameters
        ----------
        x_1 : Tensor
            Target data samples, shape (B, C, T, H, W).
        cond : Tensor, optional
            Physics parameters, shape (B, num_params).

        Returns
        -------
        FlowMatchingOutput
            Dataclass with predicted_velocity, target_velocity, x_t.
        """
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

        return FlowMatchingOutput(
            predicted_velocity=v_pred,
            target_velocity=v_target,
            x_t=x_t,
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.inference_mode()
    def sample(
        self,
        shape: tuple[int, ...],
        cond: Optional[Tensor] = None,
        num_steps: int = 100,
        method: str = "euler",
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

        Returns
        -------
        Tensor
            Generated samples at t=1, shape (B, C, T, H, W).
        """
        device = self.device
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
