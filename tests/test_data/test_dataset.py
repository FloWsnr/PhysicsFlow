"""Unit tests for PhysicsDataset class and _get_dataset function."""

import numpy as np
import torch
from typing import Callable

from physicsflow.data.dataset import PhysicsDataset, _get_dataset, compute_skip_indices
from physicsflow.data.steadystate_idx import compute_steadystate_indices
from the_well.data.normalization import RMSNormalization, ZScoreNormalization


class TestPhysicsDataset:
    """Tests for PhysicsDataset class."""

    def test_init(self, write_dummy_dataset: Callable):
        """Test PhysicsDataset initialization."""
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
        assert isinstance(dataset, PhysicsDataset)
        assert len(dataset) > 0

    def test_getitem_returns_dict(self, write_dummy_dataset: Callable):
        """Test that __getitem__ returns a dictionary with tensors."""
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

        data = dataset[0]
        assert isinstance(data, dict)
        assert "input_fields" in data
        assert "output_fields" in data
        assert isinstance(data["input_fields"], torch.Tensor)
        assert isinstance(data["output_fields"], torch.Tensor)

    def test_getitem_shape_rearrangement(self, write_dummy_dataset: Callable):
        """Test that __getitem__ correctly rearranges dimensions from (T, H, W, C) to (C, T, H, W)."""
        n_steps_input = 3
        n_steps_output = 2

        data_dir = write_dummy_dataset()
        dataset = PhysicsDataset(
            data_dir=data_dir,
            norm_path=data_dir / "stats.yaml",
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            use_normalization=False,
            normalization_type=ZScoreNormalization,
            min_dt_stride=1,
            max_dt_stride=1,
            full_trajectory_mode=False,
            max_rollout_steps=10,
        )

        data = dataset[0]
        x = data["input_fields"]
        y = data["output_fields"]

        # Check that x has shape (C, T, H, W) with T = n_steps_input
        assert x.ndim == 4, f"Expected x to have 4 dimensions, got {x.ndim}"
        assert x.shape[1] == n_steps_input, (
            f"Expected x.shape[1] == {n_steps_input}, got {x.shape[1]}"
        )

        # Check that y has shape (C, T, H, W) with T = n_steps_output
        assert y.ndim == 4, f"Expected y to have 4 dimensions, got {y.ndim}"
        assert y.shape[1] == n_steps_output, (
            f"Expected y.shape[1] == {n_steps_output}, got {y.shape[1]}"
        )

        # Check that x and y have same C, H, W dimensions
        assert x.shape[0] == y.shape[0], "x and y should have same number of channels"
        assert x.shape[2] == y.shape[2], "x and y should have same height"
        assert x.shape[3] == y.shape[3], "x and y should have same width"

    def test_with_normalization(self, write_dummy_dataset: Callable):
        """Test dataset with normalization enabled."""
        data_dir = write_dummy_dataset()
        dataset = PhysicsDataset(
            data_dir=data_dir,
            norm_path=data_dir / "stats.yaml",
            n_steps_input=2,
            n_steps_output=1,
            use_normalization=True,
            normalization_type=ZScoreNormalization,
            min_dt_stride=1,
            max_dt_stride=1,
            full_trajectory_mode=False,
            max_rollout_steps=10,
        )

        data = dataset[0]
        assert isinstance(data["input_fields"], torch.Tensor)
        assert isinstance(data["output_fields"], torch.Tensor)

    def test_different_dt_strides(self, write_dummy_dataset: Callable):
        """Test dataset with different dt strides."""
        data_dir = write_dummy_dataset()
        dataset = PhysicsDataset(
            data_dir=data_dir,
            norm_path=data_dir / "stats.yaml",
            n_steps_input=2,
            n_steps_output=1,
            use_normalization=False,
            normalization_type=ZScoreNormalization,
            min_dt_stride=1,
            max_dt_stride=3,
            full_trajectory_mode=False,
            max_rollout_steps=10,
        )

        data = dataset[0]
        assert isinstance(data["input_fields"], torch.Tensor)
        assert isinstance(data["output_fields"], torch.Tensor)

    def test_rms_normalization(self, write_dummy_dataset: Callable):
        """Test dataset with RMS normalization."""
        data_dir = write_dummy_dataset()
        dataset = PhysicsDataset(
            data_dir=data_dir,
            norm_path=data_dir / "stats.yaml",
            n_steps_input=3,
            n_steps_output=1,
            use_normalization=True,
            normalization_type=RMSNormalization,
            min_dt_stride=1,
            max_dt_stride=1,
            full_trajectory_mode=False,
            max_rollout_steps=10,
        )

        data = dataset[0]
        assert isinstance(data["input_fields"], torch.Tensor)
        assert isinstance(data["output_fields"], torch.Tensor)


class TestGetDatasetFunction:
    """Tests for _get_dataset helper function."""

    def test_get_dataset_with_fixed_stride(self, write_dummy_dataset: Callable):
        """Test _get_dataset with fixed dt_stride (int)."""
        data_dir = write_dummy_dataset()
        dataset = _get_dataset(
            data_dir=data_dir,
            norm_path=data_dir / "stats.yaml",
            n_steps_input=3,
            n_steps_output=1,
            use_normalization=False,
            normalization_type=ZScoreNormalization,
            dt_stride=2,
            full_trajectory_mode=False,
            max_rollout_steps=10,
        )

        assert isinstance(dataset, PhysicsDataset)
        data = dataset[0]
        assert isinstance(data["input_fields"], torch.Tensor)
        assert isinstance(data["output_fields"], torch.Tensor)

    def test_get_dataset_with_stride_range(self, write_dummy_dataset: Callable):
        """Test _get_dataset with dt_stride as [min, max] list."""
        data_dir = write_dummy_dataset()
        dataset = _get_dataset(
            data_dir=data_dir,
            norm_path=data_dir / "stats.yaml",
            n_steps_input=2,
            n_steps_output=1,
            use_normalization=True,
            normalization_type=ZScoreNormalization,
            dt_stride=[1, 3],
            full_trajectory_mode=False,
            max_rollout_steps=10,
        )

        assert isinstance(dataset, PhysicsDataset)
        data = dataset[0]
        assert isinstance(data["input_fields"], torch.Tensor)
        assert isinstance(data["output_fields"], torch.Tensor)

    def test_get_dataset_default_params(self, write_dummy_dataset: Callable):
        """Test _get_dataset with default parameters."""
        data_dir = write_dummy_dataset()
        dataset = _get_dataset(
            data_dir=data_dir,
            norm_path=data_dir / "stats.yaml",
        )

        assert isinstance(dataset, PhysicsDataset)
        data = dataset[0]
        x = data["input_fields"]
        y = data["output_fields"]

        # With default n_steps_input=1 and n_steps_output=1
        assert x.shape[1] == 1
        assert y.shape[1] == 1


class TestComputeSkipIndices:
    """Tests for compute_skip_indices function."""

    def test_basic_skip_computation(self):
        """Test basic skip index computation for a single trajectory."""
        # Trajectory reaches steady state at timestep 10
        steady_state_info = {"0": {"0": 10}}
        n_trajectories_per_file = [1]
        n_steps_per_trajectory = [20]
        n_steps_input = 4
        n_steps_output = 1
        dt_stride = 1

        skip_indices = compute_skip_indices(
            steady_state_info,
            n_trajectories_per_file,
            n_steps_per_trajectory,
            n_steps_input,
            n_steps_output,
            dt_stride,
        )

        # With n_steps_input=4, dt_stride=1, output starts at time_idx + 4
        # Skip if time_idx + 4 >= 10, i.e., time_idx >= 6
        # windows_per_traj = 20 - (1 + 1*(4+1-1)) + 1 = 16
        # Windows: 0-15, skip from 6 onwards: 6, 7, 8, ..., 15 (10 indices)
        assert 0 not in skip_indices
        assert 5 not in skip_indices
        assert 6 in skip_indices
        assert 15 in skip_indices
        assert len(skip_indices) == 10  # indices 6-15

    def test_no_steady_state(self):
        """Test with no steady state info (empty dict)."""
        steady_state_info = {}
        n_trajectories_per_file = [5]
        n_steps_per_trajectory = [20]
        n_steps_input = 4
        n_steps_output = 1
        dt_stride = 1

        skip_indices = compute_skip_indices(
            steady_state_info,
            n_trajectories_per_file,
            n_steps_per_trajectory,
            n_steps_input,
            n_steps_output,
            dt_stride,
        )

        assert len(skip_indices) == 0

    def test_all_from_start_steady_state(self):
        """Test when steady state starts at timestep 0."""
        # Steady state from very beginning
        steady_state_info = {"0": {"0": 0}}
        n_trajectories_per_file = [1]
        n_steps_per_trajectory = [10]
        n_steps_input = 2
        n_steps_output = 1
        dt_stride = 1

        skip_indices = compute_skip_indices(
            steady_state_info,
            n_trajectories_per_file,
            n_steps_per_trajectory,
            n_steps_input,
            n_steps_output,
            dt_stride,
        )

        # Output starts at time_idx + 2, skip if time_idx + 2 >= 0
        # That means skip if time_idx >= -2, which is always true
        # windows_per_traj = 10 - (1 + 1*(2+1-1)) + 1 = 8
        # All 8 windows should be skipped
        assert len(skip_indices) == 8
        assert list(range(8)) == skip_indices

    def test_multi_trajectory_single_file(self):
        """Test with multiple trajectories in a single file."""
        # File with 3 trajectories:
        # - Traj 0: steady at timestep 5
        # - Traj 1: no steady state (not in dict)
        # - Traj 2: steady at timestep 8
        steady_state_info = {"0": {"0": 5, "2": 8}}
        n_trajectories_per_file = [3]
        n_steps_per_trajectory = [15]
        n_steps_input = 2
        n_steps_output = 1
        dt_stride = 1

        skip_indices = compute_skip_indices(
            steady_state_info,
            n_trajectories_per_file,
            n_steps_per_trajectory,
            n_steps_input,
            n_steps_output,
            dt_stride,
        )

        # windows_per_traj = 15 - (1 + 1*(2+1-1)) + 1 = 13
        # Traj 0: skip if time_idx >= 5 - 2 = 3, windows 3-12 (10 indices)
        #   Global: 0*13 + 3 to 0*13 + 12 = 3 to 12
        # Traj 1: no steady state, no skips (global 13-25)
        # Traj 2: skip if time_idx >= 8 - 2 = 6, windows 6-12 (7 indices)
        #   Global: 2*13 + 6 to 2*13 + 12 = 32 to 38

        # Check trajectory 0 skips
        for i in range(3, 13):
            assert i in skip_indices, f"Index {i} should be skipped"
        for i in range(0, 3):
            assert i not in skip_indices, f"Index {i} should not be skipped"

        # Check trajectory 1 has no skips
        for i in range(13, 26):
            assert i not in skip_indices, f"Index {i} should not be skipped"

        # Check trajectory 2 skips
        for i in range(32, 39):
            assert i in skip_indices, f"Index {i} should be skipped"
        for i in range(26, 32):
            assert i not in skip_indices, f"Index {i} should not be skipped"

    def test_multi_file_dataset(self):
        """Test with multiple files."""
        # File 0: 2 trajectories, steady at timestep 4 for traj 0
        # File 1: 1 trajectory, steady at timestep 6 for traj 0
        steady_state_info = {
            "0": {"0": 4},
            "1": {"0": 6},
        }
        n_trajectories_per_file = [2, 1]
        n_steps_per_trajectory = [10, 12]  # Different steps per file
        n_steps_input = 2
        n_steps_output = 1
        dt_stride = 1

        skip_indices = compute_skip_indices(
            steady_state_info,
            n_trajectories_per_file,
            n_steps_per_trajectory,
            n_steps_input,
            n_steps_output,
            dt_stride,
        )

        # File 0: windows = 10 - (1 + 1*(2+1-1)) + 1 = 8
        # Traj 0: skip from time_idx >= 4-2=2, windows 2-7, global 2-7 (6 indices)
        # Traj 1: no steady state, no skips (global 8-15)
        # File 1: windows = 12 - (1 + 1*(2+1-1)) + 1 = 10
        # Traj 0: skip from time_idx >= 6-2=4, windows 4-9 (6 indices)
        #   global offset = 2*8 = 16, global 16+4=20 to 16+9=25

        # File 0 offset = 0
        # File 1 offset = 2 * 8 = 16

        # Check file 0, traj 0
        assert 0 not in skip_indices
        assert 1 not in skip_indices
        assert 2 in skip_indices
        assert 7 in skip_indices

        # Check file 0, traj 1 (no skips)
        for i in range(8, 16):
            assert i not in skip_indices

        # Check file 1, traj 0
        assert 16 not in skip_indices  # time_idx 0
        assert 19 not in skip_indices  # time_idx 3
        assert 20 in skip_indices  # time_idx 4
        assert 25 in skip_indices  # time_idx 9

    def test_larger_stride(self):
        """Test with larger dt_stride."""
        steady_state_info = {"0": {"0": 10}}
        n_trajectories_per_file = [1]
        n_steps_per_trajectory = [30]
        n_steps_input = 4
        n_steps_output = 1
        dt_stride = 2

        skip_indices = compute_skip_indices(
            steady_state_info,
            n_trajectories_per_file,
            n_steps_per_trajectory,
            n_steps_input,
            n_steps_output,
            dt_stride,
        )

        # windows = 30 - (1 + 2*(4+1-1)) + 1 = 22
        # Output starts at time_idx + 4*2 = time_idx + 8
        # Skip if time_idx + 8 >= 10, i.e., time_idx >= 2
        # Skip windows 2-21 (20 indices)

        assert 0 not in skip_indices
        assert 1 not in skip_indices
        assert 2 in skip_indices
        assert 21 in skip_indices
        assert len(skip_indices) == 20

    def test_sorted_output(self):
        """Test that skip indices are sorted."""
        # Multiple trajectories with different steady state points
        steady_state_info = {"0": {"2": 5, "0": 8, "1": 3}}
        n_trajectories_per_file = [3]
        n_steps_per_trajectory = [15]
        n_steps_input = 2
        n_steps_output = 1
        dt_stride = 1

        skip_indices = compute_skip_indices(
            steady_state_info,
            n_trajectories_per_file,
            n_steps_per_trajectory,
            n_steps_input,
            n_steps_output,
            dt_stride,
        )

        # Verify sorted
        assert skip_indices == sorted(skip_indices)

    def test_edge_case_steady_at_last_valid_window(self):
        """Test when steady state starts at the last valid window."""
        # Steady at timestep that only affects the very last few windows
        steady_state_info = {"0": {"0": 12}}
        n_trajectories_per_file = [1]
        n_steps_per_trajectory = [15]
        n_steps_input = 2
        n_steps_output = 1
        dt_stride = 1

        skip_indices = compute_skip_indices(
            steady_state_info,
            n_trajectories_per_file,
            n_steps_per_trajectory,
            n_steps_input,
            n_steps_output,
            dt_stride,
        )

        # windows = 15 - (1 + 1*(2+1-1)) + 1 = 13
        # Skip if time_idx >= 12 - 2 = 10
        # Skip windows 10, 11, 12 (3 indices)
        assert len(skip_indices) == 3
        assert 10 in skip_indices
        assert 11 in skip_indices
        assert 12 in skip_indices

    def test_with_real_dataset_and_steady_states(self, write_dummy_dataset: Callable):
        """Test compute_skip_indices with a real dataset and computed steady state info."""
        # Create dataset with known steady state characteristics
        H, W = 16, 16
        T = 15
        n_datasets = 2
        n_traj_per_file = 3
        n_steps_input = 2
        n_steps_output = 1
        dt_stride = 1

        t_ss_1 = 5
        t_ss_2 = 10

        # Generate data for 3 trajectories per file:
        # - Traj 0: steady state starting at timestep 5
        # - Traj 1: no steady state (all random)
        # - Traj 2: steady state starting at timestep 10
        t0_data = np.random.rand(n_traj_per_file, T, H, W).astype(np.float32)
        t1_data = np.random.rand(n_traj_per_file, T, H, W, 2).astype(np.float32)

        # Make traj 0 steady from timestep 5 onwards
        for t in range(t_ss_1, T):
            t0_data[0, t, :, :] = t0_data[0, t_ss_1 - 1, :, :]
            t1_data[0, t, :, :, :] = t1_data[0, t_ss_1 - 1, :, :, :]

        # Make traj 2 steady from timestep 10 onwards
        for t in range(t_ss_2, T):
            t0_data[2, t, :, :] = t0_data[2, t_ss_2 - 1, :, :]
            t1_data[2, t, :, :, :] = t1_data[2, t_ss_2 - 1, :, :, :]

        path = write_dummy_dataset(
            sub_path="skip_indices_test_data",
            n_datasets=n_datasets,
            t0_data=t0_data,
            t1_data=t1_data,
        )

        # Create dataset for steady state computation
        dataset_full_traj = PhysicsDataset(
            data_dir=path,
            norm_path=path / "stats.yaml",
            n_steps_input=1,
            n_steps_output=1,
            use_normalization=True,
            normalization_type=ZScoreNormalization,
            min_dt_stride=1,
            max_dt_stride=1,
            full_trajectory_mode=True,
            max_rollout_steps=1000,
        )

        # Compute steady state indices
        steady_state_info = compute_steadystate_indices(
            dataset=dataset_full_traj, atol=1e-2, num_workers=1
        )

        # Create dataset for steady state computation
        dataset_without_skip = PhysicsDataset(
            data_dir=path,
            norm_path=path / "stats.yaml",
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            use_normalization=True,
            normalization_type=ZScoreNormalization,
            min_dt_stride=1,
            max_dt_stride=1,
            full_trajectory_mode=False,
            max_rollout_steps=1000,
        )

        # Get dataset properties for skip computation
        n_trajectories_per_file = dataset_without_skip.n_trajectories_per_file
        n_steps_per_trajectory = dataset_without_skip.n_steps_per_trajectory

        # Convert int keys to str to match compute_skip_indices signature
        # (compute_skip_indices expects string keys since it's designed for JSON-loaded data)
        steady_state_info_str = {
            str(file_idx): {str(traj_idx): ts for traj_idx, ts in traj_dict.items()}
            for file_idx, traj_dict in steady_state_info.items()
        }

        # Compute skip indices
        skip_indices = compute_skip_indices(
            steady_state_info_str,
            [int(x) for x in n_trajectories_per_file],
            [int(x) for x in n_steps_per_trajectory],
            n_steps_input,
            n_steps_output,
            dt_stride,
        )

        # Verify the steady state detection worked as expected
        # When we make frame[t_ss:] = frame[t_ss-1], the algorithm detects steady
        # state at t_ss-1 (the first index where diff is zero in the diffs array)
        # diffs[t] = frame[t] - frame[t+1], so diffs[t_ss-1] = frame[t_ss-1] - frame[t_ss] = 0
        detected_t_ss_1 = t_ss_1 - 1  # Algorithm returns 4 for "steady from t=5 onwards"
        detected_t_ss_2 = t_ss_2 - 1  # Algorithm returns 9 for "steady from t=10 onwards"

        # Verify the detected values match what we expect
        # File 0 should have traj 0 and traj 2 with steady states
        assert steady_state_info_str.get("0") is not None, "File 0 should have steady states"
        assert steady_state_info_str["0"].get("0") == detected_t_ss_1, (
            f"Traj 0 steady state: expected {detected_t_ss_1}, got {steady_state_info_str['0'].get('0')}"
        )
        assert steady_state_info_str["0"].get("2") == detected_t_ss_2, (
            f"Traj 2 steady state: expected {detected_t_ss_2}, got {steady_state_info_str['0'].get('2')}"
        )
        # Traj 1 should NOT be in steady state info (no steady state detected)
        assert "1" not in steady_state_info_str["0"], "Traj 1 should not have steady state"

        # Verify the skip indices are correct
        # windows = T - (n_steps_input + n_steps_output) + 1 = 15 - 3 + 1 = 13 windows per traj
        windows_per_traj = T - (n_steps_input + n_steps_output) + 1
        assert windows_per_traj == 13

        # For traj 0 (detected at t_ss_1-1=4): skip if time_idx >= 4 - 2*1 = 2
        # Skip windows 2,3,4,5,6,7,8,9,10,11,12 = 11 windows
        first_skip_traj0 = max(0, detected_t_ss_1 - n_steps_input * dt_stride)
        traj0_skip_count = windows_per_traj - first_skip_traj0

        # For traj 2 (detected at t_ss_2-1=9): skip if time_idx >= 9 - 2*1 = 7
        # Skip windows 7,8,9,10,11,12 = 6 windows
        first_skip_traj2 = max(0, detected_t_ss_2 - n_steps_input * dt_stride)
        traj2_skip_count = windows_per_traj - first_skip_traj2

        # Expected skips per file: 11 + 6 = 17
        skips_per_file = traj0_skip_count + traj2_skip_count
        # Total expected: 17 * 2 = 34
        expected_total = skips_per_file * n_datasets

        assert len(skip_indices) == expected_total, (
            f"Expected {expected_total} skip indices, got {len(skip_indices)}"
        )

        # Verify skip indices are sorted
        assert skip_indices == sorted(skip_indices), "Skip indices should be sorted"

        # Verify no duplicate indices
        assert len(skip_indices) == len(set(skip_indices)), "Skip indices should be unique"

        # Verify specific indices for file 0
        # File 0, traj 0 starts at global index 0
        # Traj 0 should skip indices starting from first_skip_traj0 to windows_per_traj-1
        for time_idx in range(first_skip_traj0, windows_per_traj):
            assert time_idx in skip_indices, f"Traj 0 time_idx {time_idx} should be in skip_indices"

        # File 0, traj 2 starts at global index 2 * windows_per_traj = 26
        traj2_base = 2 * windows_per_traj
        for time_idx in range(first_skip_traj2, windows_per_traj):
            global_idx = traj2_base + time_idx
            assert global_idx in skip_indices, f"Traj 2 global_idx {global_idx} should be in skip_indices"
