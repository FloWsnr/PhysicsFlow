"""Diffusion model implementation.

Denoising Diffusion Probabilistic Models (DDPM) learn to reverse
a gradual noising process to generate samples from the data distribution.
"""

import random
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from physicsflow.models.common import sinusoidal_embedding
from physicsflow.models.diffusion.schedules import DiffusionSchedule, get_schedule


@dataclass
class DiffusionOutput:
    """Output from DiffusionModel.forward().

    Attributes
    ----------
    loss : Tensor
        Training loss.
    pred : Tensor
        Model prediction (noise, x0, or v depending on objective).
    target : Tensor
        Target for the prediction.
    """

    loss: Tensor
    pred: Tensor
    target: Tensor


def _extract(a: Tensor, t: Tensor, x_shape: tuple) -> Tensor:
    """Extract values from tensor a at indices t and reshape for broadcasting.

    Parameters
    ----------
    a : Tensor
        1D tensor of values to extract from, shape (timesteps,).
    t : Tensor
        Indices to extract, shape (batch_size,).
    x_shape : tuple
        Shape of the data tensor for broadcasting.

    Returns
    -------
    Tensor
        Extracted and reshaped values, shape (batch_size, 1, 1, ...).
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


class DiffusionModel(nn.Module):
    """Denoising Diffusion Probabilistic Model for physics simulations.

    Supports multiple beta schedules, prediction objectives, and conditioning
    on physics parameters via constant_scalars.

    Parameters
    ----------
    network : nn.Module
        Neural network that predicts noise/x0/velocity. Should accept:
        - x_t: (B, C, T, H, W) noisy samples
        - t: (B,) integer timesteps
        - cond: (B, num_params) physics parameters (optional)
        And return prediction of same shape as x_t.
    timesteps : int
        Number of diffusion timesteps.
    schedule : str
        Beta schedule type: "linear", "cosine", or "sigmoid".
    objective : str
        Prediction objective: "pred_noise", "pred_x0", or "pred_v".
    loss_type : str
        Loss function: "l1", "l2", or "huber".
    snr_weighting : bool
        Whether to use SNR-based loss weighting.
    min_snr_gamma : float
        Minimum SNR for clamping in SNR weighting.
    """

    def __init__(
        self,
        network: nn.Module,
        timesteps: int = 1000,
        schedule: Literal["linear", "cosine", "sigmoid"] = "cosine",
        objective: Literal["pred_noise", "pred_x0", "pred_v"] = "pred_noise",
        loss_type: Literal["l1", "l2", "huber"] = "l2",
        snr_weighting: bool = False,
        min_snr_gamma: float = 5.0,
        **schedule_kwargs,
    ):
        super().__init__()
        self.network = network
        self.timesteps = timesteps
        self.objective = objective
        self.loss_type = loss_type
        self.snr_weighting = snr_weighting
        self.min_snr_gamma = min_snr_gamma

        # Create diffusion schedule
        diffusion_schedule = get_schedule(schedule, timesteps, **schedule_kwargs)

        # Register schedule tensors as buffers (move to device with model)
        self._register_schedule_buffers(diffusion_schedule)

    def _register_schedule_buffers(self, schedule: DiffusionSchedule) -> None:
        """Register schedule tensors as buffers."""
        self.register_buffer("betas", schedule.betas)
        self.register_buffer("alphas_cumprod", schedule.alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", schedule.alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", schedule.sqrt_alphas_cumprod)
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", schedule.sqrt_one_minus_alphas_cumprod
        )
        self.register_buffer("sqrt_recip_alphas_cumprod", schedule.sqrt_recip_alphas_cumprod)
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", schedule.sqrt_recipm1_alphas_cumprod
        )
        self.register_buffer("posterior_variance", schedule.posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped", schedule.posterior_log_variance_clipped
        )
        self.register_buffer("posterior_mean_coef1", schedule.posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", schedule.posterior_mean_coef2)
        self.register_buffer("snr", schedule.snr)

    def q_sample(
        self,
        x_start: Tensor,
        t: Tensor,
        noise: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """Forward diffusion process: q(x_t | x_0).

        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise

        Parameters
        ----------
        x_start : Tensor
            Clean data x_0, shape (B, C, T, H, W).
        t : Tensor
            Timesteps, shape (B,).
        noise : Tensor, optional
            Pre-generated noise (default: generate new noise).

        Returns
        -------
        tuple[Tensor, Tensor]
            (x_t, noise) - noisy sample and the noise used.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = _extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = _extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        x_t = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t, noise

    def predict_start_from_noise(
        self,
        x_t: Tensor,
        t: Tensor,
        noise: Tensor,
    ) -> Tensor:
        """Predict x_0 from x_t and predicted noise.

        x_0 = (x_t - sqrt(1 - alpha_bar_t) * noise) / sqrt(alpha_bar_t)
        """
        return (
            _extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(
        self,
        x_t: Tensor,
        t: Tensor,
        x_start: Tensor,
    ) -> Tensor:
        """Predict noise from x_t and x_0."""
        return (
            _extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x_start
        ) / _extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def predict_v(
        self,
        x_start: Tensor,
        t: Tensor,
        noise: Tensor,
    ) -> Tensor:
        """Compute velocity target.

        v = sqrt(alpha_bar_t) * noise - sqrt(1 - alpha_bar_t) * x_0
        """
        return (
            _extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise
            - _extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(
        self,
        x_t: Tensor,
        t: Tensor,
        v: Tensor,
    ) -> Tensor:
        """Predict x_0 from x_t and predicted velocity."""
        return (
            _extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - _extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def model_predictions(
        self,
        x_t: Tensor,
        t: Tensor,
        cond: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """Get model predictions and convert to (pred_noise, x_start).

        Parameters
        ----------
        x_t : Tensor
            Noisy input at timestep t.
        t : Tensor
            Timesteps.
        cond : Tensor, optional
            Conditioning from constant_scalars.

        Returns
        -------
        tuple[Tensor, Tensor]
            (predicted_noise, predicted_x_start)
        """
        model_output = self.network(x_t, t, cond)

        if self.objective == "pred_noise":
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x_t, t, pred_noise)
        elif self.objective == "pred_x0":
            x_start = model_output
            pred_noise = self.predict_noise_from_start(x_t, t, x_start)
        elif self.objective == "pred_v":
            v = model_output
            x_start = self.predict_start_from_v(x_t, t, v)
            pred_noise = self.predict_noise_from_start(x_t, t, x_start)
        else:
            raise ValueError(f"Unknown objective: {self.objective}")

        return pred_noise, x_start

    def q_posterior_mean_variance(
        self,
        x_start: Tensor,
        x_t: Tensor,
        t: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute mean and variance for q(x_{t-1} | x_t, x_0)."""
        posterior_mean = (
            _extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = _extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance

    def p_mean_variance(
        self,
        x_t: Tensor,
        t: Tensor,
        cond: Optional[Tensor] = None,
        clip_denoised: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute mean and variance for p(x_{t-1} | x_t).

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            (model_mean, posterior_variance, posterior_log_variance)
        """
        _, x_start = self.model_predictions(x_t, t, cond)

        if clip_denoised:
            x_start = torch.clamp(x_start, -1.0, 1.0)

        return self.q_posterior_mean_variance(x_start, x_t, t)

    @torch.inference_mode()
    def p_sample(
        self,
        x_t: Tensor,
        t: Tensor,
        cond: Optional[Tensor] = None,
    ) -> Tensor:
        """Single denoising step: sample x_{t-1} from p(x_{t-1} | x_t)."""
        batch_size = x_t.shape[0]
        device = x_t.device

        # Create batched timestep tensor
        batched_t = torch.full((batch_size,), t, device=device, dtype=torch.long)

        model_mean, _, model_log_variance = self.p_mean_variance(
            x_t, batched_t, cond, clip_denoised=True
        )

        noise = torch.randn_like(x_t) if t > 0 else 0.0
        return model_mean + (0.5 * model_log_variance).exp() * noise

    @torch.inference_mode()
    def sample(
        self,
        shape: tuple[int, ...],
        cond: Optional[Tensor] = None,
        return_all_timesteps: bool = False,
        device: Optional[torch.device] = None,
    ) -> Tensor | list[Tensor]:
        """Generate samples via iterative denoising.

        Parameters
        ----------
        shape : tuple
            Shape of samples to generate (B, C, T, H, W).
        cond : Tensor, optional
            Conditioning from constant_scalars, shape (B, n_scalars).
        return_all_timesteps : bool
            If True, return samples at all timesteps.
        device : torch.device, optional
            Device to generate samples on.

        Returns
        -------
        Tensor or list[Tensor]
            Generated samples, shape (B, C, T, H, W).
        """
        if device is None:
            device = next(self.parameters()).device

        # Start from pure noise
        x = torch.randn(shape, device=device)

        intermediates = [x] if return_all_timesteps else None

        # Reverse diffusion
        for t in reversed(range(self.timesteps)):
            x = self.p_sample(x, t, cond)
            if return_all_timesteps:
                intermediates.append(x)

        if return_all_timesteps:
            return intermediates

        return x

    def compute_loss(
        self,
        pred: Tensor,
        target: Tensor,
        t: Tensor,
    ) -> Tensor:
        """Compute loss with optional SNR weighting.

        Parameters
        ----------
        pred : Tensor
            Model prediction.
        target : Tensor
            Target (noise, x0, or velocity).
        t : Tensor
            Timesteps (used for SNR weighting).

        Returns
        -------
        Tensor
            Scalar loss value.
        """
        # Base loss per element
        if self.loss_type == "l1":
            loss = F.l1_loss(pred, target, reduction="none")
        elif self.loss_type == "l2":
            loss = F.mse_loss(pred, target, reduction="none")
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(pred, target, reduction="none")
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Reduce spatial dimensions, keep batch dimension
        loss = loss.mean(dim=list(range(1, loss.ndim)))  # Shape: (B,)

        # SNR weighting (min-SNR-gamma strategy)
        if self.snr_weighting:
            snr_t = _extract(self.snr, t, (t.shape[0], 1))[:, 0]  # Shape: (B,)
            snr_clipped = snr_t.clamp(max=self.min_snr_gamma)

            if self.objective == "pred_noise":
                weight = snr_clipped / snr_t
            elif self.objective == "pred_x0":
                weight = snr_clipped
            elif self.objective == "pred_v":
                weight = snr_clipped / (snr_t + 1)
            else:
                weight = torch.ones_like(snr_t)

            loss = loss * weight

        return loss.mean()

    def forward(self, data: dict[str, Tensor]) -> DiffusionOutput:
        """Training forward pass.

        Samples random timesteps, adds noise, predicts, and computes loss.

        Parameters
        ----------
        data : dict
            Dictionary containing:
            - 'input_fields': Clean data x_start, shape (B, C, T, H, W)
            - 'constant_scalars': Physics parameters, shape (B, num_params)

        Returns
        -------
        DiffusionOutput
            Dataclass containing loss, prediction, and target.
        """
        x_start = data["input_fields"]
        cond = data.get("constant_scalars", None)

        batch_size = x_start.shape[0]
        device = x_start.device

        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()

        # Generate noise
        noise = torch.randn_like(x_start)

        # Forward diffusion
        x_t, noise = self.q_sample(x_start, t, noise)

        # Get target based on objective
        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_start
        elif self.objective == "pred_v":
            target = self.predict_v(x_start, t, noise)
        else:
            raise ValueError(f"Unknown objective: {self.objective}")

        # Model prediction
        pred = self.network(x_t, t, cond)

        # Compute loss
        loss = self.compute_loss(pred, target, t)

        return DiffusionOutput(loss=loss, pred=pred, target=target)


class PlaceholderDenoiser(nn.Module):
    """Placeholder 1-layer MLP denoiser network for testing.

    This is a simple implementation to verify the diffusion
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
        """Predict noise/x0/velocity.

        Parameters
        ----------
        x_t : Tensor
            Noisy samples, shape (B, C, T, H, W).
        t : Tensor
            Integer timesteps, shape (B,).
        cond : Tensor, optional
            Physics parameters, shape (B, cond_dim).

        Returns
        -------
        Tensor
            Predicted noise/x0/velocity, shape (B, C, T, H, W).
        """
        batch_size = x_t.shape[0]
        original_shape = x_t.shape

        # Flatten spatial dimensions
        x_flat = x_t.reshape(batch_size, -1)

        # Sinusoidal time embedding (normalize timesteps to [0, 1] range)
        t_normalized = t.float() / 1000.0  # Assuming max 1000 timesteps
        t_emb = sinusoidal_embedding(t_normalized, self.time_embed_dim)

        # Concatenate inputs
        if cond is not None:
            inputs = torch.cat([x_flat, t_emb, cond], dim=-1)
        else:
            inputs = torch.cat([x_flat, t_emb], dim=-1)

        # Forward through network
        output = self.net(inputs)

        # Reshape to original spatial dimensions
        return output.reshape(*original_shape)
