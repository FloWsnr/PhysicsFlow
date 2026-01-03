"""Tests for dataloader module.

By: Claude Code
Date: 2025-10-16
"""

from typing import Callable

import torch
import pytest
from the_well.data.normalization import ZScoreNormalization
from torch.utils.data import RandomSampler, SequentialSampler

from physicsflow.data.dataloader import get_dataloader, _downsample_spatial, collate_fn
from physicsflow.data.dataset import PhysicsDataset


@pytest.fixture(scope="session")
def dataset(write_dummy_dataset: Callable):
    """Fixture that provides a PhysicsDataset instance.

    The default return shape of input and output fields is
    (C, T, H, W) = (6, 2, 32, 32) and (6, 1, 32, 32) respectively.
    """
    data_dir = write_dummy_dataset()
    dataset = PhysicsDataset(
        data_dir=data_dir,
        norm_path=data_dir / "stats.yaml",
        n_steps_input=2,
        n_steps_output=1,
        use_normalization=False,
        normalization_type=ZScoreNormalization,
        min_dt_stride=1,
        max_dt_stride=1,
        full_trajectory_mode=False,
        max_rollout_steps=10,
    )
    return dataset


class TestGetDataloader:
    def test_get_dataloader_random_sampler(self, dataset: PhysicsDataset):
        """Test dataloader with random sampler."""

        dataloader = get_dataloader(
            dataset=dataset,
            seed=42,
            batch_size=16,
            num_workers=0,
            is_distributed=False,
            shuffle=True,
        )

        assert dataloader.batch_size == 16
        assert isinstance(dataloader.sampler, RandomSampler)
        assert dataloader.pin_memory is True
        assert dataloader.drop_last is True

    def test_get_dataloader_sequential_sampler(self, dataset: PhysicsDataset):
        """Test dataloader with sequential sampler."""
        dataloader = get_dataloader(
            dataset=dataset,
            seed=42,
            batch_size=16,
            num_workers=0,
            is_distributed=False,
            shuffle=False,
        )

        assert dataloader.batch_size == 16
        assert isinstance(dataloader.sampler, SequentialSampler)
        assert dataloader.pin_memory is True
        assert dataloader.drop_last is True

    def test_get_dataloader_iteration(self, dataset: PhysicsDataset):
        """Test that dataloader can be iterated."""

        dataloader = get_dataloader(
            dataset=dataset,
            seed=42,
            batch_size=16,
            num_workers=0,
            is_distributed=False,
            shuffle=False,
        )

        # Iterate through dataloader
        for batch in dataloader:
            assert batch["input_fields"].shape == (16, 6, 2, 32, 32)
            assert batch["output_fields"].shape == (16, 6, 1, 32, 32)

    def test_get_dataloader_seed_consistency(self, dataset: PhysicsDataset):
        """Test that same seed produces same order."""

        # Create two dataloaders with same seed
        dataloader1 = get_dataloader(
            dataset=dataset,
            seed=42,
            batch_size=16,
            num_workers=0,
            is_distributed=False,
            shuffle=True,
        )

        dataloader2 = get_dataloader(
            dataset=dataset,
            seed=42,
            batch_size=16,
            num_workers=0,
            is_distributed=False,
            shuffle=True,
        )

        # Get first batch from each
        batch1 = next(iter(dataloader1))
        batch2 = next(iter(dataloader2))

        # They should be equal (same indices sampled)
        assert torch.allclose(batch1["input_fields"], batch2["input_fields"])
        assert torch.allclose(batch1["output_fields"], batch2["output_fields"])

    def test_get_dataloader_different_seeds(self, dataset: PhysicsDataset):
        """Test that different seeds produce different orders."""

        # Create two dataloaders with different seeds
        dataloader1 = get_dataloader(
            dataset=dataset,
            seed=42,
            batch_size=16,
            num_workers=0,
            is_distributed=False,
            shuffle=True,
        )

        dataloader2 = get_dataloader(
            dataset=dataset,
            seed=123,
            batch_size=16,
            num_workers=0,
            is_distributed=False,
            shuffle=True,
        )

        # Get first batch from each
        batch1 = next(iter(dataloader1))
        batch2 = next(iter(dataloader2))

        # They should be different (different indices sampled)
        assert not torch.allclose(batch1["input_fields"], batch2["input_fields"])


class TestDownsampleSpatial:
    def test_downsample_spatial_bilinear(self):
        """Test spatial downsampling with bilinear interpolation."""
        # Create input tensor (B, C, T, H, W)
        B, C, T, H, W = 2, 3, 4, 64, 64
        data = torch.randn(B, C, T, H, W)

        # Downsample to 32x32
        target_size = (32, 32)
        downsampled = _downsample_spatial(data, target_size, "bilinear")

        # Check output shape
        assert downsampled.shape == (B, C, T, 32, 32)

    def test_downsample_spatial_nearest(self):
        """Test spatial downsampling with nearest neighbor interpolation."""
        B, C, T, H, W = 2, 3, 4, 64, 64
        data = torch.randn(B, C, T, H, W)

        target_size = (32, 32)
        downsampled = _downsample_spatial(data, target_size, "nearest")

        assert downsampled.shape == (B, C, T, 32, 32)

    def test_downsample_spatial_preserves_temporal(self):
        """Test that downsampling preserves temporal dimension."""
        B, C, T, H, W = 1, 2, 10, 64, 64
        data = torch.randn(B, C, T, H, W)

        target_size = (16, 16)
        downsampled = _downsample_spatial(data, target_size, "bilinear")

        # Temporal dimension should be unchanged
        assert downsampled.shape[2] == T

    def test_downsample_spatial_different_sizes(self):
        """Test downsampling to non-square sizes."""
        B, C, T, H, W = 2, 3, 4, 128, 64
        data = torch.randn(B, C, T, H, W)

        # Downsample to non-square size
        target_size = (32, 16)
        downsampled = _downsample_spatial(data, target_size, "bilinear")

        assert downsampled.shape == (B, C, T, 32, 16)

    def test_downsample_spatial_upsampling(self):
        """Test that function can also upsample (though named downsample)."""
        B, C, T, H, W = 1, 2, 3, 16, 16
        data = torch.randn(B, C, T, H, W)

        # Upsample to larger size
        target_size = (32, 32)
        upsampled = _downsample_spatial(data, target_size, "bilinear")

        assert upsampled.shape == (B, C, T, 32, 32)

    def test_downsample_spatial_values_reasonable(self):
        """Test that downsampling produces reasonable values."""
        B, C, T, H, W = 1, 1, 1, 4, 4
        # Create a simple pattern
        data = torch.ones(B, C, T, H, W)

        # Downsample
        target_size = (2, 2)
        downsampled = _downsample_spatial(data, target_size, "bilinear")

        # All values should still be 1 (averaging ones gives ones)
        assert torch.allclose(downsampled, torch.ones_like(downsampled))


class TestCollateFn:
    def test_collate_fn_basic(self):
        """Test collate function with spatial downsampling."""
        # Create a batch of samples (dictionaries with C, T, H, W tensors)
        C, T, H, W = 2, 4, 64, 64
        batch = [
            {
                "input_fields": torch.randn(C, T, H, W),
                "output_fields": torch.randn(C, T, H, W),
            },
            {
                "input_fields": torch.randn(C, T, H, W),
                "output_fields": torch.randn(C, T, H, W),
            },
        ]

        target_size = (32, 32)
        result = collate_fn(batch, target_size, "bilinear")

        # Check shapes
        assert result["input_fields"].shape == (2, C, T, 32, 32)
        assert result["output_fields"].shape == (2, C, T, 32, 32)

    def test_collate_fn_different_modes(self):
        """Test collate function with different interpolation modes."""
        C, T, H, W = 2, 4, 64, 64
        batch = [
            {
                "input_fields": torch.randn(C, T, H, W),
                "output_fields": torch.randn(C, T, H, W),
            },
        ]

        target_size = (32, 32)

        # Test different modes
        for mode in ["bilinear", "nearest", "bicubic", "area"]:
            result = collate_fn(batch, target_size, mode)
            assert result["input_fields"].shape == (1, C, T, 32, 32)
            assert result["output_fields"].shape == (1, C, T, 32, 32)


class TestGetDataloaderWithDownsampling:
    def test_get_dataloader_with_downsampling(self, dataset: PhysicsDataset):
        """Test dataloader with spatial downsampling."""

        target_size = (16, 16)
        dataloader = get_dataloader(
            dataset=dataset,
            seed=42,
            batch_size=2,
            num_workers=0,
            is_distributed=False,
            shuffle=False,
            spatial_downsample_size=target_size,
            downsample_mode="bilinear",
        )

        # Get a batch
        batch = next(iter(dataloader))

        # Check that downsampling was applied
        assert batch["input_fields"].shape == (2, 6, 2, 16, 16)
        assert batch["output_fields"].shape == (2, 6, 1, 16, 16)

    def test_get_dataloader_downsample_modes(self, dataset: PhysicsDataset):
        """Test dataloader with different downsampling modes."""
        target_size = (16, 16)

        for mode in ["bilinear", "nearest", "bicubic"]:
            dataloader = get_dataloader(
                dataset=dataset,
                seed=42,
                batch_size=2,
                num_workers=0,
                is_distributed=False,
                shuffle=False,
                spatial_downsample_size=target_size,
                downsample_mode=mode,
            )

            batch = next(iter(dataloader))
            assert batch["input_fields"].shape == (2, 6, 2, 16, 16)
            assert batch["output_fields"].shape == (2, 6, 1, 16, 16)
