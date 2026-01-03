"""Diffusion noise schedules.

Beta schedules define how noise is added during the forward diffusion process.
Different schedules affect training dynamics and sample quality.
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class DiffusionSchedule:
    """Precomputed diffusion schedule values.

    All tensors have shape (timesteps,) and are used for efficient
    computation during training and sampling.

    Attributes
    ----------
    betas : Tensor
        Noise schedule beta_t values.
    alphas : Tensor
        1 - beta_t values.
    alphas_cumprod : Tensor
        Cumulative product of alphas (alpha_bar_t).
    alphas_cumprod_prev : Tensor
        alphas_cumprod shifted by 1 (with 1.0 prepended).
    sqrt_alphas_cumprod : Tensor
        sqrt(alpha_bar_t) - coefficient for x_0 in forward process.
    sqrt_one_minus_alphas_cumprod : Tensor
        sqrt(1 - alpha_bar_t) - coefficient for noise in forward process.
    sqrt_recip_alphas_cumprod : Tensor
        1/sqrt(alpha_bar_t) - used in x_0 prediction from noise.
    sqrt_recipm1_alphas_cumprod : Tensor
        sqrt(1/alpha_bar_t - 1) - used in x_0 prediction from noise.
    posterior_variance : Tensor
        Variance for q(x_{t-1}|x_t, x_0).
    posterior_log_variance_clipped : Tensor
        Clipped log posterior variance.
    posterior_mean_coef1 : Tensor
        Coefficient for x_0 in posterior mean.
    posterior_mean_coef2 : Tensor
        Coefficient for x_t in posterior mean.
    snr : Tensor
        Signal-to-noise ratio at each timestep.
    """

    betas: Tensor
    alphas: Tensor
    alphas_cumprod: Tensor
    alphas_cumprod_prev: Tensor
    sqrt_alphas_cumprod: Tensor
    sqrt_one_minus_alphas_cumprod: Tensor
    sqrt_recip_alphas_cumprod: Tensor
    sqrt_recipm1_alphas_cumprod: Tensor
    posterior_variance: Tensor
    posterior_log_variance_clipped: Tensor
    posterior_mean_coef1: Tensor
    posterior_mean_coef2: Tensor
    snr: Tensor


def linear_beta_schedule(
    timesteps: int,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
) -> Tensor:
    """Linear beta schedule from original DDPM paper.

    Parameters
    ----------
    timesteps : int
        Number of diffusion timesteps.
    beta_start : float
        Starting beta value.
    beta_end : float
        Ending beta value.

    Returns
    -------
    Tensor
        Beta values, shape (timesteps,).
    """
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> Tensor:
    """Cosine beta schedule from 'Improved DDPM' paper.

    Uses alpha_bar(t) = f(t)/f(0), where f(t) = cos((t/T + s)/(1+s) * pi/2)^2

    Parameters
    ----------
    timesteps : int
        Number of diffusion timesteps.
    s : float
        Small offset to prevent singularity near t=0.

    Returns
    -------
    Tensor
        Beta values, shape (timesteps,).
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


def sigmoid_beta_schedule(
    timesteps: int,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
) -> Tensor:
    """Sigmoid beta schedule for improved performance on larger images.

    Parameters
    ----------
    timesteps : int
        Number of diffusion timesteps.
    beta_start : float
        Starting beta value.
    beta_end : float
        Ending beta value.

    Returns
    -------
    Tensor
        Beta values, shape (timesteps,).
    """
    betas = torch.linspace(-6, 6, timesteps)
    betas = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
    return betas


def _compute_schedule(betas: Tensor) -> DiffusionSchedule:
    """Compute all derived schedule values from betas.

    Parameters
    ----------
    betas : Tensor
        Beta values, shape (timesteps,).

    Returns
    -------
    DiffusionSchedule
        Dataclass with all precomputed values.
    """
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

    # Forward process coefficients
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    # Coefficients for predicting x_0 from noise
    sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
    sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1)

    # Posterior q(x_{t-1} | x_t, x_0) coefficients
    posterior_variance = (
        betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    )
    # Clip to avoid log(0)
    posterior_log_variance_clipped = torch.log(
        torch.clamp(posterior_variance, min=1e-20)
    )

    posterior_mean_coef1 = (
        betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    )
    posterior_mean_coef2 = (
        (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
    )

    # Signal-to-noise ratio
    snr = alphas_cumprod / (1.0 - alphas_cumprod)

    return DiffusionSchedule(
        betas=betas,
        alphas=alphas,
        alphas_cumprod=alphas_cumprod,
        alphas_cumprod_prev=alphas_cumprod_prev,
        sqrt_alphas_cumprod=sqrt_alphas_cumprod,
        sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
        sqrt_recip_alphas_cumprod=sqrt_recip_alphas_cumprod,
        sqrt_recipm1_alphas_cumprod=sqrt_recipm1_alphas_cumprod,
        posterior_variance=posterior_variance,
        posterior_log_variance_clipped=posterior_log_variance_clipped,
        posterior_mean_coef1=posterior_mean_coef1,
        posterior_mean_coef2=posterior_mean_coef2,
        snr=snr,
    )


def get_schedule(
    schedule_type: str,
    timesteps: int,
    **kwargs,
) -> DiffusionSchedule:
    """Create a diffusion schedule with all precomputed values.

    Parameters
    ----------
    schedule_type : str
        Type of schedule: "linear", "cosine", or "sigmoid".
    timesteps : int
        Number of diffusion timesteps.
    **kwargs
        Additional arguments passed to the schedule function.

    Returns
    -------
    DiffusionSchedule
        Dataclass containing all precomputed schedule values.

    Raises
    ------
    ValueError
        If schedule_type is not recognized.
    """
    schedule_fns = {
        "linear": linear_beta_schedule,
        "cosine": cosine_beta_schedule,
        "sigmoid": sigmoid_beta_schedule,
    }

    if schedule_type not in schedule_fns:
        raise ValueError(
            f"Unknown schedule: {schedule_type}. "
            f"Available: {list(schedule_fns.keys())}"
        )

    betas = schedule_fns[schedule_type](timesteps, **kwargs)
    return _compute_schedule(betas)
