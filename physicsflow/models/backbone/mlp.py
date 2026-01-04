"""Simple MLP backbone for testing generative models.

This module provides a placeholder MLP network that can be used with both
diffusion and flow matching models for testing and prototyping.
"""

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from physicsflow.models.common import sinusoidal_embedding


class MLPBackbone(nn.Module):
    """Simple MLP backbone for generative models.

    This is a placeholder implementation to verify the diffusion/flow matching
    infrastructure works correctly. Replace with a proper architecture
    (e.g., U-Net, DiT) for actual training.

    Works with both diffusion models (integer timesteps) and flow matching
    (continuous time in [0, 1]).

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
        """Predict output (noise, velocity, or x0 depending on training objective).

        Parameters
        ----------
        x_t : Tensor
            Noised samples, shape (B, C, T, H, W).
        t : Tensor
            Time values, shape (B,).
            Can be integer timesteps (diffusion) or floats in [0, 1] (flow matching).
        cond : Tensor, optional
            Physics parameters, shape (B, cond_dim).

        Returns
        -------
        Tensor
            Predicted output, shape (B, C, T, H, W).
        """
        batch_size = x_t.shape[0]
        original_shape = x_t.shape

        # Flatten spatial dimensions
        x_flat = x_t.reshape(batch_size, -1)

        # Normalize time to [0, 1] range for embedding
        # Handle both integer timesteps and continuous time
        t_float = t.float()
        if t_float.max() > 1.0:
            # Integer timesteps - normalize (assume max 1000)
            t_float = t_float / 1000.0

        # Sinusoidal time embedding
        t_emb = sinusoidal_embedding(t_float, self.time_embed_dim)

        # Concatenate inputs
        if cond is not None:
            inputs = torch.cat([x_flat, t_emb, cond], dim=-1)
        else:
            inputs = torch.cat([x_flat, t_emb], dim=-1)

        # Forward through network
        output = self.net(inputs)

        # Reshape to original spatial dimensions
        return output.reshape(*original_shape)
