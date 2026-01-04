"""Simple MLP backbone for testing generative models.

This module provides a placeholder MLP network that can be used with
flow matching models for testing and prototyping.
"""

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from physicsflow.models.common import sinusoidal_embedding


class MLPBackbone(nn.Module):
    """Simple MLP backbone for generative models.

    This is a placeholder implementation to verify the flow matching
    infrastructure works correctly. Replace with a proper architecture
    (e.g., U-Net, DiT) for actual training.

    Works with continuous time in [0, 1] for flow matching.

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
        """Predict velocity for flow matching.

        Parameters
        ----------
        x_t : Tensor
            Interpolated samples, shape (B, C, T, H, W).
        t : Tensor
            Time values in [0, 1], shape (B,).
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

        # Time is expected in [0, 1] for flow matching
        t_float = t.float()

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
