"""Tests for steady state index computation utilities."""

from typing import Callable
from unittest.mock import patch

import numpy as np

from physicsflow.data.steadystate_idx import (
    _check_steady_state_index,
    _find_first_consecutive_run,
    compute_steadystate_indices,
)
from physicsflow.data.dataset import PhysicsDataset
from the_well.data.normalization import ZScoreNormalization


class TestCheckSteadyStateIndex:
    """Tests for _check_steady_state_index function."""

    def test_steady_state_returns_tuple(self, write_dummy_dataset: Callable):
        """Test that steady state data returns (traj_idx, first_steady_timestep) tuple."""

        # Create matching first and last frames
        H, W = 16, 16
        T = 10
        n_datasets = 2
        n_traj_per_file = 2
        n_traj = n_traj_per_file * n_datasets
        t0_data = np.random.rand(n_traj_per_file, H, W).astype(np.float32)
        t1_data = np.random.rand(n_traj_per_file, H, W, 2).astype(np.float32)

        # Expand temporal dimension: use np.tile to repeat along new axis
        t0_data = np.tile(t0_data[:, None, :, :], (1, T, 1, 1))  # (n_traj, T=10, H, W)
        t1_data = np.tile(
            t1_data[:, None, :, :, :], (1, T, 1, 1, 1)
        )  # (n_traj, T=10, H, W, C)

        path = write_dummy_dataset(
            sub_path="steady_state_data",
            t0_data=t0_data,
            t1_data=t1_data,
        )

        dataset = PhysicsDataset(
            data_dir=path,
            norm_path=path / "stats.yaml",
            n_steps_input=2,
            n_steps_output=1,
            use_normalization=True,
            normalization_type=ZScoreNormalization,
            min_dt_stride=1,
            max_dt_stride=1,
            full_trajectory_mode=True,
            max_rollout_steps=100,
        )

        traj_idx = 0
        atol = 1e-2
        min_consecutive = 1
        args = (dataset, traj_idx, atol, min_consecutive)

        result = _check_steady_state_index(args)
        # Should return (traj_idx, first_steady_timestep)
        assert result is not None
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] == traj_idx
        # First steady timestep is 0 since all frames are identical
        assert result[1] == 0

    def test_non_steady_state_returns_none(self, write_dummy_dataset: Callable):
        """Test that non-steady state data returns None."""
        # Create dataset with no steady state frames (all random)
        H, W = 16, 16
        T = 10
        n_traj = 2

        # Generate completely random data (no steady state)
        t0_data = np.random.rand(n_traj, T, H, W).astype(np.float32)
        t1_data = np.random.rand(n_traj, T, H, W, 2).astype(np.float32)

        path = write_dummy_dataset(
            sub_path="non_steady_data",
            n_datasets=1,
            t0_data=t0_data,
            t1_data=t1_data,
        )

        dataset = PhysicsDataset(
            data_dir=path,
            norm_path=path / "stats.yaml",
            n_steps_input=2,
            n_steps_output=1,
            use_normalization=True,
            normalization_type=ZScoreNormalization,
            min_dt_stride=1,
            max_dt_stride=1,
            full_trajectory_mode=True,
            max_rollout_steps=10,
        )
        traj_idx = 0
        atol = 1e-6  # Strict tolerance
        min_consecutive = 1
        args = (dataset, traj_idx, atol, min_consecutive)

        result = _check_steady_state_index(args)
        assert result is None

    def test_tolerance_threshold(self, write_dummy_dataset: Callable):
        """Test that tolerance threshold works correctly."""
        # Create steady state data with small deviations (tol=5e-3)
        H, W = 16, 16
        T = 10
        n_traj = 2
        n_steady_state = 5

        # Generate data with last n_steady_state frames having small deviations
        t0_data = np.random.rand(n_traj, T, H, W).astype(np.float32)
        t1_data = np.random.rand(n_traj, T, H, W, 2).astype(np.float32)

        # Make last n_steady_state frames nearly identical with small noise
        for i in range(n_traj):
            base_frame_t0 = t0_data[i, -n_steady_state, :, :]
            base_frame_t1 = t1_data[i, -n_steady_state, :, :, :]
            for j in range(1, n_steady_state):
                t0_data[i, -n_steady_state + j, :, :] = (
                    base_frame_t0 + np.random.randn(H, W).astype(np.float32) * 5e-3
                )
                t1_data[i, -n_steady_state + j, :, :, :] = (
                    base_frame_t1 + np.random.randn(H, W, 2).astype(np.float32) * 5e-3
                )

        path = write_dummy_dataset(
            sub_path="tolerance_data",
            n_datasets=1,
            t0_data=t0_data,
            t1_data=t1_data,
        )

        dataset = PhysicsDataset(
            data_dir=path,
            norm_path=path / "stats.yaml",
            n_steps_input=2,
            n_steps_output=1,
            use_normalization=True,
            normalization_type=ZScoreNormalization,
            min_dt_stride=1,
            max_dt_stride=1,
            full_trajectory_mode=True,
            max_rollout_steps=100,
        )

        traj_idx = 0
        min_consecutive = 1

        # With strict tolerance, should return None (difference > tolerance)
        args_strict = (dataset, traj_idx, 1e-3, min_consecutive)
        result_strict = _check_steady_state_index(args_strict)
        assert result_strict is None

        # With loose tolerance, should return tuple
        args_loose = (dataset, traj_idx, 1e-1, min_consecutive)
        result_loose = _check_steady_state_index(args_loose)
        assert result_loose is not None
        assert result_loose[0] == traj_idx
        # Should detect steady state in the last 5 frames
        assert result_loose[1] > 0

    def test_partial_steady_state(self, write_dummy_dataset: Callable):
        """Test trajectory that becomes steady after some timesteps."""
        # Create dataset where last 3 frames are steady state
        # dim_t = 10, so frames 0-6 are random, frames 7-9 are steady
        H, W = 16, 16
        T = 10
        n_traj = 1
        n_steady_state = 3

        # Generate random data
        t0_data = np.random.rand(n_traj, T, H, W).astype(np.float32)
        t1_data = np.random.rand(n_traj, T, H, W, 2).astype(np.float32)

        # Make last n_steady_state frames identical
        for i in range(n_traj):
            steady_frame_t0 = t0_data[i, -n_steady_state, :, :]
            steady_frame_t1 = t1_data[i, -n_steady_state, :, :, :]
            for j in range(1, n_steady_state):
                t0_data[i, -n_steady_state + j, :, :] = steady_frame_t0
                t1_data[i, -n_steady_state + j, :, :, :] = steady_frame_t1

        path = write_dummy_dataset(
            sub_path="partial_steady_data",
            n_datasets=1,
            t0_data=t0_data,
            t1_data=t1_data,
        )

        dataset = PhysicsDataset(
            data_dir=path,
            norm_path=path / "stats.yaml",
            n_steps_input=2,
            n_steps_output=1,
            use_normalization=True,
            normalization_type=ZScoreNormalization,
            min_dt_stride=1,
            max_dt_stride=1,
            full_trajectory_mode=True,
            max_rollout_steps=100,
        )

        traj_idx = 0
        atol = 1e-2
        min_consecutive = 1
        args = (dataset, traj_idx, atol, min_consecutive)

        result = _check_steady_state_index(args)
        assert result is not None
        assert result[0] == traj_idx
        # Steady state should be detected in the last few frames
        # Since we have 3 steady state frames at the end, it should be > 0
        assert result[1] > 0

    def test_error_handling_returns_none(self):
        """Test that exceptions during processing return None."""

        class ErrorDataset:
            """Dataset that raises an exception when accessed."""

            def __getitem__(self, _idx):
                raise Exception("Test error")

            def __len__(self):
                return 1

        dataset = ErrorDataset()
        traj_idx = 0
        atol = 1e-2
        min_consecutive = 1
        args = (dataset, traj_idx, atol, min_consecutive)

        # Should not raise, should return None and log warning
        with patch("physicsflow.data.steadystate_idx.logging.warning") as mock_warning:
            result = _check_steady_state_index(args)
            assert result is None
            mock_warning.assert_called_once()

    def test_missing_fields_returns_none(self):
        """Test that missing fields in dataset return None."""

        # Create a mock dataset that returns invalid data
        class InvalidDataset:
            def __getitem__(self, idx):
                return {}  # Missing 'input_fields' and 'output_fields'

            def __len__(self):
                return 1

        dataset = InvalidDataset()
        traj_idx = 0
        atol = 1e-2
        min_consecutive = 1
        args = (dataset, traj_idx, atol, min_consecutive)

        result = _check_steady_state_index(args)
        assert result is None

    def test_min_consecutive_requirement(self, write_dummy_dataset: Callable):
        """Test that min_consecutive parameter correctly requires N consecutive steady steps."""
        # Create dataset where all frames from index 5 onward are identical.
        # This gives us a long run of steady state diffs starting at diff index 5.
        # With min_consecutive=3: should return index 5
        # With min_consecutive=5: should also return index 5
        H, W = 16, 16
        T = 12
        n_traj = 1
        steady_start = 5  # Make frames 5..11 all identical

        # Start with random data (use large values to ensure distinct from steady)
        t0_data = np.random.rand(n_traj, T, H, W).astype(np.float32) * 100
        t1_data = np.random.rand(n_traj, T, H, W, 2).astype(np.float32) * 100

        # Make all frames from steady_start onward identical
        for i in range(steady_start + 1, T):
            t0_data[0, i] = t0_data[0, steady_start]
            t1_data[0, i] = t1_data[0, steady_start]

        path = write_dummy_dataset(
            sub_path="min_consecutive_data",
            n_datasets=1,
            t0_data=t0_data,
            t1_data=t1_data,
        )

        dataset = PhysicsDataset(
            data_dir=path,
            norm_path=path / "stats.yaml",
            n_steps_input=1,
            n_steps_output=1,
            use_normalization=False,  # Disable normalization to preserve exact values
            normalization_type=ZScoreNormalization,
            min_dt_stride=1,
            max_dt_stride=1,
            full_trajectory_mode=True,
            max_rollout_steps=100,
        )

        traj_idx = 0
        atol = 1e-2

        # With min_consecutive=1, should find first steady at index 5
        args_1 = (dataset, traj_idx, atol, 1)
        result_1 = _check_steady_state_index(args_1)
        assert result_1 is not None
        assert result_1[0] == traj_idx
        assert result_1[1] == steady_start

        # With min_consecutive=3, should still find at index 5 (we have 6 consecutive)
        args_3 = (dataset, traj_idx, atol, 3)
        result_3 = _check_steady_state_index(args_3)
        assert result_3 is not None
        assert result_3[0] == traj_idx
        assert result_3[1] == steady_start

        # With min_consecutive=6, should still find at index 5 (we have exactly 6)
        args_6 = (dataset, traj_idx, atol, 6)
        result_6 = _check_steady_state_index(args_6)
        assert result_6 is not None
        assert result_6[0] == traj_idx
        assert result_6[1] == steady_start

        # With min_consecutive=7, should find no run (we only have 6)
        args_7 = (dataset, traj_idx, atol, 7)
        result_7 = _check_steady_state_index(args_7)
        assert result_7 is None


class TestFindFirstConsecutiveRun:
    """Tests for _find_first_consecutive_run helper function."""

    def test_all_true(self):
        """Test with all True values."""
        import torch

        mask = torch.tensor([True, True, True, True, True])
        assert _find_first_consecutive_run(mask, 3) == 0
        assert _find_first_consecutive_run(mask, 5) == 0
        assert _find_first_consecutive_run(mask, 1) == 0

    def test_all_false(self):
        """Test with all False values."""
        import torch

        mask = torch.tensor([False, False, False, False, False])
        assert _find_first_consecutive_run(mask, 1) is None
        assert _find_first_consecutive_run(mask, 3) is None

    def test_run_at_start(self):
        """Test with consecutive run at the start."""
        import torch

        mask = torch.tensor([True, True, True, False, False])
        assert _find_first_consecutive_run(mask, 3) == 0
        assert _find_first_consecutive_run(mask, 4) is None

    def test_run_at_end(self):
        """Test with consecutive run at the end."""
        import torch

        mask = torch.tensor([False, False, True, True, True])
        assert _find_first_consecutive_run(mask, 3) == 2
        assert _find_first_consecutive_run(mask, 4) is None

    def test_run_in_middle(self):
        """Test with consecutive run in the middle."""
        import torch

        mask = torch.tensor([False, True, True, True, False])
        assert _find_first_consecutive_run(mask, 3) == 1
        assert _find_first_consecutive_run(mask, 4) is None

    def test_multiple_runs_returns_first(self):
        """Test that first qualifying run is returned when multiple exist."""
        import torch

        mask = torch.tensor([True, True, False, True, True, True])
        assert _find_first_consecutive_run(mask, 2) == 0  # First run of 2 at index 0
        assert _find_first_consecutive_run(mask, 3) == 3  # First run of 3 at index 3

    def test_scattered_trues(self):
        """Test with scattered True values that don't form required run."""
        import torch

        mask = torch.tensor([True, False, True, False, True, False, True])
        assert _find_first_consecutive_run(mask, 1) == 0
        assert _find_first_consecutive_run(mask, 2) is None

    def test_min_length_equals_mask_length(self):
        """Test when min_length equals mask length."""
        import torch

        mask = torch.tensor([True, True, True])
        assert _find_first_consecutive_run(mask, 3) == 0

        mask = torch.tensor([True, True, False])
        assert _find_first_consecutive_run(mask, 3) is None

    def test_min_length_exceeds_mask_length(self):
        """Test when min_length exceeds mask length."""
        import torch

        mask = torch.tensor([True, True])
        assert _find_first_consecutive_run(mask, 5) is None

    def test_invalid_min_length_raises(self):
        """Test that invalid min_length raises ValueError."""
        import torch
        import pytest

        mask = torch.tensor([True, True, True])
        with pytest.raises(ValueError):
            _find_first_consecutive_run(mask, 0)
        with pytest.raises(ValueError):
            _find_first_consecutive_run(mask, -1)

    def test_empty_mask(self):
        """Test with empty mask."""
        import torch

        mask = torch.tensor([], dtype=torch.bool)
        assert _find_first_consecutive_run(mask, 1) is None


class TestComputeSteadystateIndices:
    """Tests for compute_steadystate_indices function."""

    def test_all_steady_state(self, write_dummy_dataset: Callable):
        """Test with dataset where all trajectories reach steady state immediately."""
        # Create dataset with all frames in steady state
        H, W = 16, 16
        T = 10
        n_traj_per_file = 5

        # Generate data where all frames are identical (perfect steady state)
        t0_data = np.random.rand(n_traj_per_file, 1, H, W).astype(np.float32)
        t1_data = np.random.rand(n_traj_per_file, 1, H, W, 2).astype(np.float32)

        # Repeat same frame for all timesteps
        t0_data = np.tile(t0_data, (1, T, 1, 1))
        t1_data = np.tile(t1_data, (1, T, 1, 1, 1))

        path = write_dummy_dataset(
            sub_path="all_steady_data",
            n_datasets=1,
            t0_data=t0_data,
            t1_data=t1_data,
        )

        dataset = PhysicsDataset(
            data_dir=path,
            norm_path=path / "stats.yaml",
            n_steps_input=2,
            n_steps_output=1,
            use_normalization=True,
            normalization_type=ZScoreNormalization,
            min_dt_stride=1,
            max_dt_stride=1,
            full_trajectory_mode=True,
            max_rollout_steps=100,
        )

        result = compute_steadystate_indices(dataset=dataset, atol=1e-2, num_workers=2)

        # Result should be nested dict: {file_idx: {traj_idx: first_steady_timestep}}
        assert isinstance(result, dict)
        assert 0 in result  # File 0
        assert len(result[0]) == n_traj_per_file  # 5 trajectories with steady states
        # Each trajectory has steady state starting at timestep 0
        for traj_idx in range(n_traj_per_file):
            assert traj_idx in result[0]
            assert result[0][traj_idx] == 0

    def test_no_steady_state(self, write_dummy_dataset: Callable):
        """Test with dataset where no trajectories reach steady state."""
        # Create dataset with no steady state frames (all random)
        H, W = 16, 16
        T = 10
        n_traj_per_file = 3

        # Generate completely random data (no steady state)
        t0_data = np.random.rand(n_traj_per_file, T, H, W).astype(np.float32)
        t1_data = np.random.rand(n_traj_per_file, T, H, W, 2).astype(np.float32)

        path = write_dummy_dataset(
            sub_path="no_steady_data",
            n_datasets=1,
            t0_data=t0_data,
            t1_data=t1_data,
        )

        dataset = PhysicsDataset(
            data_dir=path,
            norm_path=path / "stats.yaml",
            n_steps_input=2,
            n_steps_output=1,
            use_normalization=True,
            normalization_type=ZScoreNormalization,
            min_dt_stride=1,
            max_dt_stride=1,
            full_trajectory_mode=True,
            max_rollout_steps=100,
        )

        result = compute_steadystate_indices(dataset=dataset, atol=1e-6, num_workers=2)

        # Result should be empty dict
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_mixed_steady_and_non_steady(self, write_dummy_dataset: Callable):
        """Test with dataset containing trajectories with different steady state characteristics."""
        # Create dataset with partial steady state
        H, W = 16, 16
        T = 10
        n_traj_per_file = 6
        n_steady_state = 5

        # Generate data with last n_steady_state frames having small deviations
        t0_data = np.random.rand(n_traj_per_file, T, H, W).astype(np.float32)
        t1_data = np.random.rand(n_traj_per_file, T, H, W, 2).astype(np.float32)

        # Make last n_steady_state frames nearly identical with small noise
        for i in range(n_traj_per_file // 2):
            base_frame_t0 = t0_data[i, -n_steady_state, :, :]
            base_frame_t1 = t1_data[i, -n_steady_state, :, :, :]
            for j in range(1, n_steady_state):
                t0_data[i, -n_steady_state + j, :, :] = (
                    base_frame_t0 + np.random.randn(H, W).astype(np.float32) * 1e-2
                )
                t1_data[i, -n_steady_state + j, :, :, :] = (
                    base_frame_t1 + np.random.randn(H, W, 2).astype(np.float32) * 1e-2
                )

        path = write_dummy_dataset(
            sub_path="mixed_data",
            n_datasets=1,
            t0_data=t0_data,
            t1_data=t1_data,
        )

        dataset = PhysicsDataset(
            data_dir=path,
            norm_path=path / "stats.yaml",
            n_steps_input=4,
            n_steps_output=1,
            use_normalization=True,
            normalization_type=ZScoreNormalization,
            min_dt_stride=1,
            max_dt_stride=1,
            full_trajectory_mode=True,
            max_rollout_steps=100,
        )

        # Use a moderate tolerance that should detect the steady states
        result = compute_steadystate_indices(dataset=dataset, atol=5e-2, num_workers=2)
        # At least some trajectories should have steady states detected
        for file_idx, traj_dict in result.items():
            for i in range(n_traj_per_file // 2):
                assert i in traj_dict
                timestep = traj_dict[i]
                assert timestep >= 0

        # The other half should not have steady states
        for i in range(n_traj_per_file // 2, n_traj_per_file):
            for file_idx, traj_dict in result.items():
                assert i not in traj_dict

    def test_single_worker(self, write_dummy_dataset: Callable):
        """Test with single worker (num_workers=1)."""
        # Create dataset with all steady state frames
        H, W = 16, 16
        T = 10
        n_traj = 2

        # Generate data where all frames are identical (perfect steady state)
        t0_data = np.random.rand(n_traj, 1, H, W).astype(np.float32)
        t1_data = np.random.rand(n_traj, 1, H, W, 2).astype(np.float32)

        # Repeat same frame for all timesteps
        t0_data = np.tile(t0_data, (1, T, 1, 1))
        t1_data = np.tile(t1_data, (1, T, 1, 1, 1))

        path = write_dummy_dataset(
            sub_path="single_worker_data",
            n_datasets=1,
            t0_data=t0_data,
            t1_data=t1_data,
        )

        dataset = PhysicsDataset(
            data_dir=path,
            norm_path=path / "stats.yaml",
            n_steps_input=2,
            n_steps_output=1,
            use_normalization=True,
            normalization_type=ZScoreNormalization,
            min_dt_stride=1,
            max_dt_stride=1,
            full_trajectory_mode=True,
            max_rollout_steps=100,
        )

        result = compute_steadystate_indices(dataset=dataset, atol=1e-2, num_workers=1)

        assert isinstance(result, dict)
        assert 0 in result
        assert len(result[0]) == 2
        assert result[0][0] == 0
        assert result[0][1] == 0

    def test_empty_dataset(self):
        """Test with empty dataset."""

        class EmptyDataset:
            def __len__(self):
                return 0

            @property
            def n_trajectories_per_file(self):
                return []

        dataset = EmptyDataset()

        result = compute_steadystate_indices(dataset=dataset, atol=1e-2, num_workers=2)

        assert isinstance(result, dict)
        assert len(result) == 0

    def test_different_tolerances(self, write_dummy_dataset: Callable):
        """Test that different tolerances produce different results."""
        # Create dataset with steady state frames that have small deviations
        H, W = 16, 16
        T = 10
        n_traj = 1
        n_steady_state = 5

        # Generate data with last n_steady_state frames having small deviations
        t0_data = np.random.rand(n_traj, T, H, W).astype(np.float32)
        t1_data = np.random.rand(n_traj, T, H, W, 2).astype(np.float32)

        # Make last n_steady_state frames nearly identical with small noise
        for i in range(n_traj):
            base_frame_t0 = t0_data[i, -n_steady_state, :, :]
            base_frame_t1 = t1_data[i, -n_steady_state, :, :, :]
            for j in range(1, n_steady_state):
                t0_data[i, -n_steady_state + j, :, :] = (
                    base_frame_t0 + np.random.randn(H, W).astype(np.float32) * 5e-3
                )
                t1_data[i, -n_steady_state + j, :, :, :] = (
                    base_frame_t1 + np.random.randn(H, W, 2).astype(np.float32) * 5e-3
                )

        path = write_dummy_dataset(
            sub_path="tolerance_test_data",
            n_datasets=1,
            t0_data=t0_data,
            t1_data=t1_data,
        )

        dataset = PhysicsDataset(
            data_dir=path,
            norm_path=path / "stats.yaml",
            n_steps_input=2,
            n_steps_output=1,
            use_normalization=True,
            normalization_type=ZScoreNormalization,
            min_dt_stride=1,
            max_dt_stride=1,
            full_trajectory_mode=True,
            max_rollout_steps=100,
        )

        # Strict tolerance should find no steady states
        result_strict = compute_steadystate_indices(
            dataset=dataset, atol=1e-3, num_workers=1
        )
        assert len(result_strict) == 0

        # Loose tolerance should find steady state
        result_loose = compute_steadystate_indices(
            dataset=dataset, atol=0.1, num_workers=1
        )
        assert 0 in result_loose
        assert 0 in result_loose[0]
