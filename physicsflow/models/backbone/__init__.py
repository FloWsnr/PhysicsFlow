"""Backbone networks for generative models.

This module provides shared neural network architectures that can be used
by both diffusion and flow matching models.
"""

from physicsflow.models.backbone.mlp import MLPBackbone

__all__ = [
    "MLPBackbone",
]
