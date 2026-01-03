import logging
import os
import json

from pathlib import Path
from typing import Callable, Optional

import torch
from einops import rearrange
from the_well.data.normalization import RMSNormalization, ZScoreNormalization
from the_well.data.utils import raw_steps_to_possible_sample_t0s

from physicsflow.data.well_dataset import WellDataset


class PhysicsDataset(WellDataset):
    def __init__(
        self,
        data_dir: Path,
        norm_path: Path,
        n_steps_input: int,  # control how many steps get in x
        n_steps_output: int,  # don't need for recon. just dynamics
        use_normalization: bool,
        normalization_type: Callable,
        min_dt_stride: int,
        max_dt_stride: int,
        full_trajectory_mode: bool,
        max_rollout_steps: int,
        restrict_indices: Optional[list[int]] = None,
    ):
        super().__init__(
            path=str(data_dir),
            normalization_path=(str(norm_path)),
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            use_normalization=use_normalization,
            normalization_type=normalization_type,
            min_dt_stride=min_dt_stride,
            max_dt_stride=max_dt_stride,
            full_trajectory_mode=full_trajectory_mode,
            max_rollout_steps=max_rollout_steps,
            restrict_indices=restrict_indices,
        )

    def __getitem__(self, index) -> dict[str, torch.Tensor]:
        well_dict, metadata = super().__getitem__(index)

        well_dict["input_fields"] = rearrange(
            well_dict["input_fields"], "T H W C -> C T H W"
        )  # C (channels), T (n_timesteps), H (height), W (width)
        well_dict["output_fields"] = rearrange(
            well_dict["output_fields"], "T H W C -> C T H W"
        )
        start_time = well_dict["input_time_grid"][-1]
        scalars = well_dict["constant_scalars"]  # shape (num_scalars,)
        well_dict["constant_scalars"] = torch.cat(
            [scalars, start_time.unsqueeze(0)], dim=0
        )

        return well_dict


def compute_skip_indices(
    steady_state_info: dict[str, dict[str, int]],
    n_trajectories_per_file: list[int],
    n_steps_per_trajectory: list[int],
    n_steps_input: int,
    n_steps_output: int,
    dt_stride: int,
) -> list[int]:
    """
    Compute global indices to skip based on steady state information.

    A sample should be skipped if its OUTPUT timesteps >= first_steady_timestep.
    Output starts at: time_idx + n_steps_input * dt_stride
    So skip if: time_idx >= first_steady_timestep - n_steps_input * dt_stride

    Parameters
    ----------
    steady_state_info : dict[str, dict[str, int]]
        Nested dict: {file_idx_str: {traj_idx_str: first_steady_timestep}}
        Keys are strings (from JSON), values are ints.
    n_trajectories_per_file : list[int]
        Number of trajectories in each file.
    n_steps_per_trajectory : list[int]
        Number of timesteps in trajectories for each file.
    n_steps_input : int
        Number of input timesteps.
    n_steps_output : int
        Number of output timesteps.
    dt_stride : int
        Time stride between samples.

    Returns
    -------
    list[int]
        List of global indices to skip.
    """
    skip_indices = []

    # Compute windows per trajectory for each file
    windows_per_traj = []
    for steps in n_steps_per_trajectory:
        windows = raw_steps_to_possible_sample_t0s(
            steps, n_steps_input, n_steps_output, dt_stride
        )
        windows_per_traj.append(windows)

    # Compute global index offsets
    # file_index_offsets[i] = starting global index for file i
    file_index_offsets = [0]
    for n_traj, windows in zip(n_trajectories_per_file, windows_per_traj):
        file_index_offsets.append(file_index_offsets[-1] + n_traj * windows)

    for file_idx_str, traj_dict in steady_state_info.items():
        file_idx = int(file_idx_str)
        windows = windows_per_traj[file_idx]

        for traj_idx_str, first_steady_t in traj_dict.items():
            traj_idx = int(traj_idx_str)

            # Compute first time_idx that should be skipped
            # Skip if output timesteps >= first_steady_t
            # Output starts at: time_idx + n_steps_input * dt_stride
            # So skip if: time_idx + n_steps_input * dt_stride >= first_steady_t
            # Which means: time_idx >= first_steady_t - n_steps_input * dt_stride
            first_skip_time_idx = max(0, first_steady_t - n_steps_input * dt_stride)

            # Compute global indices for this trajectory
            base_idx = file_index_offsets[file_idx] + traj_idx * windows

            for time_idx in range(first_skip_time_idx, windows):
                global_idx = base_idx + time_idx
                skip_indices.append(global_idx)

    return sorted(skip_indices)


def get_dataset(config: dict, split: str) -> PhysicsDataset:
    """
    Get a PhysicsDataset from config.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    split : str
        Dataset split (train, valid, test).

    Returns
    -------
    PhysicsDataset
        The PhysicsDataset instance.
    """
    # Mapping from string to normalization callable
    NORMALIZATION_MAP = {
        "zscore": ZScoreNormalization,
        "rms": RMSNormalization,
    }

    data_dir: str = os.environ.get("DATA_DIR")  # type: ignore
    name: str = config.get("name")  # type: ignore
    path = Path(data_dir) / name / f"data/{split}"
    norm_path = path.parents[1] / "stats.yaml"

    n_steps_input: int = config.get("n_steps_input")  # type: ignore
    n_steps_output: int = config.get("n_steps_output")  # type: ignore
    use_normalization: bool = config.get("use_normalization", True)
    normalization_type_str: str = config.get("normalization_type", "zscore")
    dt_stride: int | list[int] = config.get("dt_stride", 1)
    full_trajectory_mode: bool = config.get("full_trajectory_mode", False)
    max_rollout_steps: int = config.get("max_rollout_steps", 10000)
    skip_steady_states: bool = config.get("skip_steady_states", False)

    # Get the normalization callable from the string
    normalization_type = NORMALIZATION_MAP.get(normalization_type_str.lower())
    if normalization_type is None:
        raise ValueError(
            f"Unknown normalization type: {normalization_type_str}. "
            f"Available options: {list(NORMALIZATION_MAP.keys())}"
        )

    restrict_indices = None
    if skip_steady_states:
        # Load per-trajectory steady state info
        indices_path = path.parents[1] / "steady_state_info.json"

        if not indices_path.exists():
            raise FileNotFoundError(
                f"Steady state info file not found at {indices_path}. "
                f"Please run 'python physicsflow/data/steadystate_idx.py "
                f"--dataset_name {name}' to precompute the steady state information."
            )

        with open(indices_path, "r") as f:
            all_steady_state_info = json.load(f)

        if split not in all_steady_state_info:
            raise KeyError(
                f"Split '{split}' not found in {indices_path}. "
                f"Available splits: {list(all_steady_state_info.keys())}. "
                f"Please run the precomputation script for this split."
            )

        steady_state_info = all_steady_state_info[split]

        # Create a temporary dataset to get metadata for skip index computation
        # We need n_trajectories_per_file and n_steps_per_trajectory
        temp_dataset = _get_dataset(
            data_dir=path,
            norm_path=norm_path,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            use_normalization=False,  # Don't need normalization for metadata
            normalization_type=normalization_type,
            dt_stride=dt_stride,
            full_trajectory_mode=False,
            max_rollout_steps=max_rollout_steps,
            restrict_indices=None,
        )

        # Use min_dt_stride for conservative exclusion
        if isinstance(dt_stride, list):
            min_stride = dt_stride[0]
        else:
            min_stride = dt_stride

        restrict_indices = compute_skip_indices(
            steady_state_info=steady_state_info,
            n_trajectories_per_file=temp_dataset.n_trajectories_per_file,
            n_steps_per_trajectory=temp_dataset.n_steps_per_trajectory,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            dt_stride=min_stride,
        )

        logging.info(
            f"Computed {len(restrict_indices)} skip indices for split '{split}' "
            f"with n_steps_input={n_steps_input}, n_steps_output={n_steps_output} "
            f"from steady state info"
        )

    return _get_dataset(
        data_dir=path,
        norm_path=norm_path,
        n_steps_input=n_steps_input,
        n_steps_output=n_steps_output,
        use_normalization=use_normalization,
        normalization_type=normalization_type,
        dt_stride=dt_stride,
        full_trajectory_mode=full_trajectory_mode,
        max_rollout_steps=max_rollout_steps,
        restrict_indices=restrict_indices,
    )


def _get_dataset(
    data_dir: Path,
    norm_path: Path,
    n_steps_input: int = 1,
    n_steps_output: int = 1,
    use_normalization: bool = True,
    normalization_type: Callable = ZScoreNormalization,
    dt_stride: int | list[int] = 1,
    full_trajectory_mode: bool = False,
    max_rollout_steps: int = 10000,
    restrict_indices: Optional[list[int]] = None,
) -> PhysicsDataset:
    """
    Get a WellDataset.

    Parameters
    ----------
    data_dir : Path
        Path to the dataset directory.
    n_steps_input : int
        Number of input time steps.
    n_steps_output : int
        Number of output time steps.
    use_normalization : bool
        Whether to use normalization.
    normalization_type : Callable
        Normalization class to use (e.g., ZScoreNormalization, RMSNormalization).
    dt_stride : int or list of int
        If int, fixed stride between time steps.
        If list of int, [min_stride, max_stride] for random stride sampling.
    full_trajectory_mode : bool
        Whether to use full trajectory mode.
    max_rollout_steps : int
        Maximum number of rollout steps for full trajectory mode.
    restrict_indices : Optional[list[int]]
        List of indices to restrict the dataset to.
    """
    if isinstance(dt_stride, list):
        min_dt_stride = dt_stride[0]
        max_dt_stride = dt_stride[1]
    else:
        min_dt_stride = dt_stride
        max_dt_stride = dt_stride

    if not (
        isinstance(normalization_type, type)
        and issubclass(normalization_type, (RMSNormalization, ZScoreNormalization))
    ):
        raise ValueError(
            "normalization_type must be RMSNormalization or ZScoreNormalization class"
        )

    return PhysicsDataset(
        data_dir=data_dir,
        norm_path=norm_path,
        n_steps_input=n_steps_input,
        n_steps_output=n_steps_output,
        use_normalization=use_normalization,
        normalization_type=normalization_type,  # type: ignore
        min_dt_stride=min_dt_stride,
        max_dt_stride=max_dt_stride,
        full_trajectory_mode=full_trajectory_mode,
        max_rollout_steps=max_rollout_steps,
        restrict_indices=restrict_indices,
    )
