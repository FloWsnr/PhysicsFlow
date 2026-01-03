"""DataLoader for the WellDatasets.

By: Florian Wiesner
Date: 2025-09-11
"""

from functools import partial
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    DistributedSampler,
    default_collate,
)
from einops import rearrange

from physicsflow.data.dataset import PhysicsDataset


def _downsample_spatial(
    data: torch.Tensor,
    spatial_downsample_size: tuple[int, ...],
    downsample_mode: str,
) -> torch.Tensor:
    """
    Downsample spatial dimensions (H, W) of the data.

    Parameters
    ----------
    data : torch.Tensor
        Input tensor of shape (B, C, T, H, W).

    spatial_downsample_size : tuple[int, ...]
        Target spatial size (H', W').

    downsample_mode : str
        Interpolation mode. Options: 'bilinear', 'nearest', 'bicubic', 'area', 'nearest-exact'.
        Default is 'bilinear'.

    Returns
    -------
    torch.Tensor
        Downsampled tensor of shape (B, T, H', W', C).
    """
    # Convert from (B, C, T, H, W) to (B, C*T, H, W) for 2D interpolation
    B, C, T, H, W = data.shape
    data = rearrange(data, "b c t h w -> b (c t) h w")

    data = F.interpolate(
        data,
        size=spatial_downsample_size,
        mode=downsample_mode,
    )

    # Reshape back to (B, C, T, H', W')
    data = rearrange(data, "b (t c) h w -> b c t h w", t=T, c=C)
    return data


def collate_fn(
    batch: list[dict[str, torch.Tensor]],
    spatial_downsample_size: tuple[int, ...],
    downsample_mode: str,
) -> dict[str, torch.Tensor]:
    """Collate function to handle batches with different sizes."""
    batched_dict = default_collate(batch)
    x = batched_dict["input_fields"]  # shape (B, C, T, H, W)
    y = batched_dict["output_fields"]  # shape (B, C, T, H, W)
    x = _downsample_spatial(
        x,
        spatial_downsample_size=spatial_downsample_size,
        downsample_mode=downsample_mode,
    )
    y = _downsample_spatial(
        y,
        spatial_downsample_size=spatial_downsample_size,
        downsample_mode=downsample_mode,
    )
    batched_dict["input_fields"] = x
    batched_dict["output_fields"] = y

    return batched_dict


def get_dataloader(
    dataset: PhysicsDataset,
    seed: int,
    batch_size: int,
    num_workers: int,
    is_distributed: bool = False,
    shuffle: bool = True,
    spatial_downsample_size: Optional[tuple[int, ...]] = None,
    downsample_mode: str = "bilinear",
) -> DataLoader:
    """Get a dataloader for the dataset.

    Parameters
    ----------
    dataset : WellDataset
        Dataset to load.
    seed : int
        Seed for the dataset.
    batch_size : int
        Batch size.
    num_workers : int
        Number of workers.
    is_distributed : bool
        Whether to use distributed sampling
    shuffle : bool
        Whether to shuffle the dataset
    spatial_downsample_size : tuple[int, ...] or None
        Target spatial size (H, W) for downsampling. If None, no downsampling is applied.
    downsample_mode : str
        Interpolation mode for downsampling.
        Options: 'bilinear', 'nearest', 'bicubic', 'area', 'nearest-exact'.
        Default is 'bilinear'.
    """

    if is_distributed:
        sampler = DistributedSampler(dataset, seed=seed, shuffle=shuffle)
    else:
        if shuffle:
            generator = torch.Generator()
            generator.manual_seed(seed)
            sampler = RandomSampler(dataset, generator=generator)
        else:
            sampler = SequentialSampler(dataset)

    if spatial_downsample_size is None:
        collate = default_collate
    else:
        collate = partial(
            collate_fn,
            spatial_downsample_size=spatial_downsample_size,
            downsample_mode=downsample_mode,
        )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
        collate_fn=collate,
        drop_last=True,
    )

    return dataloader
