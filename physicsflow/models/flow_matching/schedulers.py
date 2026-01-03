"""Flow matching schedulers defining interpolation paths.

Schedulers define how to interpolate between noise (x_0) and data (x_1):
    x_t = alpha_t * x_1 + sigma_t * x_0

The velocity field (training target) is the time derivative:
    v_t = d_alpha_t * x_1 + d_sigma_t * x_0
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class SchedulerOutput:
    """Output from scheduler at time t.

    Attributes
    ----------
    alpha_t : Tensor
        Coefficient for data (x_1).
    sigma_t : Tensor
        Coefficient for noise (x_0).
    d_alpha_t : Tensor
        Time derivative of alpha_t.
    d_sigma_t : Tensor
        Time derivative of sigma_t.
    """

    alpha_t: Tensor
    sigma_t: Tensor
    d_alpha_t: Tensor
    d_sigma_t: Tensor


class BaseScheduler(ABC):
    """Abstract base class for flow matching schedulers.

    Schedulers define the interpolation path between x_0 (noise) and x_1 (data):
        x_t = alpha_t * x_1 + sigma_t * x_0

    The velocity field (target for training) is:
        v_t = d_alpha_t * x_1 + d_sigma_t * x_0
    """

    @abstractmethod
    def __call__(self, t: Tensor) -> SchedulerOutput:
        """Compute scheduler coefficients at time t.

        Parameters
        ----------
        t : Tensor
            Time values in [0, 1], shape (batch_size,).

        Returns
        -------
        SchedulerOutput
            Dataclass containing alpha_t, sigma_t, d_alpha_t, d_sigma_t.
        """
        pass

    def sample_path(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        """Sample point on the interpolation path.

        Parameters
        ----------
        x_0 : Tensor
            Source distribution samples (noise), shape (B, ...).
        x_1 : Tensor
            Target distribution samples (data), shape (B, ...).
        t : Tensor
            Time values, shape (B,).

        Returns
        -------
        Tensor
            Interpolated samples x_t, same shape as x_0/x_1.
        """
        out = self(t)
        alpha_t = self._expand_like(out.alpha_t, x_0)
        sigma_t = self._expand_like(out.sigma_t, x_0)
        return alpha_t * x_1 + sigma_t * x_0

    def target_velocity(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        """Compute target velocity for training.

        Parameters
        ----------
        x_0 : Tensor
            Source distribution samples (noise).
        x_1 : Tensor
            Target distribution samples (data).
        t : Tensor
            Time values.

        Returns
        -------
        Tensor
            Target velocity v_t = d_alpha_t * x_1 + d_sigma_t * x_0.
        """
        out = self(t)
        d_alpha_t = self._expand_like(out.d_alpha_t, x_0)
        d_sigma_t = self._expand_like(out.d_sigma_t, x_0)
        return d_alpha_t * x_1 + d_sigma_t * x_0

    @staticmethod
    def _expand_like(t: Tensor, x: Tensor) -> Tensor:
        """Expand time tensor to match data dimensions for broadcasting."""
        while t.dim() < x.dim():
            t = t.unsqueeze(-1)
        return t


class CondOTScheduler(BaseScheduler):
    """Conditional Optimal Transport scheduler (linear interpolation).

    alpha_t = t, sigma_t = 1 - t

    This is the default OT path used in original Flow Matching.
    Provides straight-line interpolation between noise and data.
    """

    def __call__(self, t: Tensor) -> SchedulerOutput:
        return SchedulerOutput(
            alpha_t=t,
            sigma_t=1.0 - t,
            d_alpha_t=torch.ones_like(t),
            d_sigma_t=-torch.ones_like(t),
        )


class CosineScheduler(BaseScheduler):
    """Cosine scheduler.

    alpha_t = sin(pi/2 * t), sigma_t = cos(pi/2 * t)

    Provides smooth transitions at t=0 and t=1 with zero velocity
    at the boundaries. Preserves unit variance along the path when
    x_0 and x_1 have unit variance.
    """

    def __call__(self, t: Tensor) -> SchedulerOutput:
        half_pi = math.pi / 2
        alpha_t = torch.sin(half_pi * t)
        sigma_t = torch.cos(half_pi * t)
        d_alpha_t = half_pi * torch.cos(half_pi * t)
        d_sigma_t = -half_pi * torch.sin(half_pi * t)
        return SchedulerOutput(
            alpha_t=alpha_t,
            sigma_t=sigma_t,
            d_alpha_t=d_alpha_t,
            d_sigma_t=d_sigma_t,
        )


class LinearVPScheduler(BaseScheduler):
    """Linear Variance Preserving scheduler.

    alpha_t = t, sigma_t = sqrt(1 - t^2)

    Preserves variance along the path: ||x_t||^2 = const when
    x_0 and x_1 have the same variance and are uncorrelated.
    """

    def __call__(self, t: Tensor) -> SchedulerOutput:
        sigma_t = torch.sqrt(1.0 - t**2 + 1e-8)
        # d_sigma_t = -t / sqrt(1 - t^2)
        d_sigma_t = -t / sigma_t
        return SchedulerOutput(
            alpha_t=t,
            sigma_t=sigma_t,
            d_alpha_t=torch.ones_like(t),
            d_sigma_t=d_sigma_t,
        )


class PolynomialScheduler(BaseScheduler):
    """Polynomial scheduler.

    alpha_t = t^n, sigma_t = (1-t)^n

    Parameters
    ----------
    n : float
        Polynomial exponent, default 2.0.
        - n=1 gives CondOT (linear)
        - n>1 gives slower start and faster end
        - n<1 gives faster start and slower end
    """

    def __init__(self, n: float = 2.0):
        self.n = n

    def __call__(self, t: Tensor) -> SchedulerOutput:
        n = self.n
        alpha_t = t**n
        sigma_t = (1.0 - t) ** n
        # Handle edge cases for derivatives
        d_alpha_t = n * torch.pow(t + 1e-8, n - 1)
        d_sigma_t = -n * torch.pow(1.0 - t + 1e-8, n - 1)
        return SchedulerOutput(
            alpha_t=alpha_t,
            sigma_t=sigma_t,
            d_alpha_t=d_alpha_t,
            d_sigma_t=d_sigma_t,
        )


def get_scheduler(scheduler_type: str, **kwargs) -> BaseScheduler:
    """Factory function to create schedulers.

    Parameters
    ----------
    scheduler_type : str
        One of: 'cond_ot', 'cosine', 'linear_vp', 'polynomial'.
    **kwargs : dict
        Additional arguments for the scheduler (e.g., n for polynomial).

    Returns
    -------
    BaseScheduler
        The scheduler instance.

    Raises
    ------
    ValueError
        If scheduler_type is not recognized.
    """
    schedulers = {
        "cond_ot": CondOTScheduler,
        "cosine": CosineScheduler,
        "linear_vp": LinearVPScheduler,
        "polynomial": PolynomialScheduler,
    }
    if scheduler_type not in schedulers:
        raise ValueError(
            f"Unknown scheduler: {scheduler_type}. "
            f"Available: {list(schedulers.keys())}"
        )
    return schedulers[scheduler_type](**kwargs)
