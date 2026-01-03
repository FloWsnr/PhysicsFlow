"""Flow Matching models for physics simulation generation.

This module implements Flow Matching, a simulation-free approach to
training continuous normalizing flows for generative modeling.
"""

from physicsflow.models.flow_matching.schedulers import (
    SchedulerOutput,
    BaseScheduler,
    CondOTScheduler,
    CosineScheduler,
    LinearVPScheduler,
    PolynomialScheduler,
    get_scheduler,
)
from physicsflow.models.flow_matching.flow_matching_model import (
    FlowMatchingOutput,
    FlowMatchingModel,
    PlaceholderVelocityNet,
)

__all__ = [
    # Schedulers
    "SchedulerOutput",
    "BaseScheduler",
    "CondOTScheduler",
    "CosineScheduler",
    "LinearVPScheduler",
    "PolynomialScheduler",
    "get_scheduler",
    # Model
    "FlowMatchingOutput",
    "FlowMatchingModel",
    "PlaceholderVelocityNet",
]
