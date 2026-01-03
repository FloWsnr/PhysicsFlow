"""Data module for PhysicsFlow."""

from physicsflow.data.dataset import (
    PhysicsDataset,
    get_dataset,
    _get_dataset,
    compute_skip_indices,
)
from physicsflow.data.dataloader import get_dataloader, collate_fn, _downsample_spatial
from physicsflow.data.well_dataset import WellDataset, DeltaWellDataset, WellMetadata
from physicsflow.data.steadystate_idx import (
    compute_steadystate_indices,
    _check_steady_state_index,
    _find_first_consecutive_run,
)

__all__ = [
    # Dataset classes
    "PhysicsDataset",
    "WellDataset",
    "DeltaWellDataset",
    "WellMetadata",
    # Dataset functions
    "get_dataset",
    "_get_dataset",
    "compute_skip_indices",
    # Dataloader functions
    "get_dataloader",
    "collate_fn",
    "_downsample_spatial",
    # Steady state utilities
    "compute_steadystate_indices",
    "_check_steady_state_index",
    "_find_first_consecutive_run",
]
