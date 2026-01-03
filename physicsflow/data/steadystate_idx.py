"""
Utility functions for data handling in physicsflow.

This script can be run standalone to precompute steady state indices:
    python physicsflow/data/steadystate_idx.py --dataset_name <name> --n_steps_input 4 --n_steps_output 1 --atol 1e-2 --num_workers 8
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path

import torch
import numpy as np
from physicsflow.data.dataset import _get_dataset
from the_well.data.normalization import ZScoreNormalization


def _check_steady_state_index(args) -> tuple[int, int] | None:
    """
    Check if a trajectory reaches steady state.

    Helper function for parallel processing. Uses full_trajectory_mode to access
    all timesteps in a single trajectory.

    Parameters
    ----------
    args : tuple
        (dataset, traj_idx, atol, min_consecutive) tuple where dataset is in full_trajectory_mode

    Returns
    -------
    tuple[int, int] or None
        Returns (traj_idx, first_steady_timestep) if steady state is found, None otherwise.
        first_steady_timestep is the first timestep of a run of at least min_consecutive
        consecutive steady state steps.
    """
    dataset, traj_idx, atol, min_consecutive = args
    try:
        well_dict = dataset[traj_idx]
        input_fields = well_dict["input_fields"]  # shape (C, 1, H, W)
        # output is full trajectory
        output_fields = well_dict["output_fields"]  # shape (C, T, H, W)
        fields = torch.cat([input_fields, output_fields], dim=1)  # shape (C, T+1, H, W)
        diffs = fields[:, :-1, :, :] - fields[:, 1:, :, :]  # shape (C, T, H, W)
        steadystates = diffs.abs() <= atol  # shape (C, T, H, W)
        # For each timestep, check if all channels and all spatial locations are steady
        per_timestep_steady = steadystates.all(dim=(0, 2, 3))  # shape (T,)

        # Find the first run of min_consecutive consecutive steady state steps
        first_steady_t = _find_first_consecutive_run(per_timestep_steady, min_consecutive)

        if first_steady_t is None:
            logging.info(f"{traj_idx} has no {min_consecutive} consecutive steady-state steps")
            return None

        logging.info(f"{traj_idx} has {min_consecutive} consecutive steady-state steps starting at {first_steady_t}")
        return (traj_idx, first_steady_t)
    except Exception as e:
        logging.warning(f"Error processing trajectory {traj_idx}: {e}")
        return None


def _find_first_consecutive_run(mask: torch.Tensor, min_length: int) -> int | None:
    """
    Find the first index where a run of at least min_length consecutive True values starts.

    Parameters
    ----------
    mask : torch.Tensor
        1D boolean tensor
    min_length : int
        Minimum number of consecutive True values required

    Returns
    -------
    int or None
        Start index of the first run of min_length consecutive True values, or None if not found.
    """
    if min_length <= 0:
        raise ValueError("min_length must be positive")

    if len(mask) < min_length:
        return None

    # Use convolution to find runs: convolve with a kernel of ones
    # A sum of min_length means all min_length values are True
    mask_float = mask.float()
    kernel = torch.ones(min_length, device=mask.device)
    # Use conv1d: need shape (1, 1, T) for input and (1, 1, min_length) for kernel
    conv_result = torch.nn.functional.conv1d(
        mask_float.unsqueeze(0).unsqueeze(0),
        kernel.unsqueeze(0).unsqueeze(0),
        padding=0
    ).squeeze()  # shape (T - min_length + 1,)

    # Find where the sum equals min_length (all True in that window)
    valid_starts = torch.where(conv_result == min_length)[0]

    if len(valid_starts) == 0:
        return None

    return int(valid_starts[0].item())


def compute_steadystate_indices(
    dataset, atol: float = 1e-2, num_workers: int = 8, min_consecutive: int = 1
) -> dict[int, dict[int, int]]:
    """
    Compute steady state indices in parallel using multiprocessing.

    Parameters
    ----------
    dataset
        PhysicsDataset instance in full_trajectory_mode to check for steady states.
        Must have `n_trajectories_per_file` attribute for multi-file support.
    atol : float
        Absolute tolerance for determining steady state.
    num_workers : int
        Number of parallel workers to use.
    min_consecutive : int
        Minimum number of consecutive steady state steps required.
        Default is 1 (original behavior). Set to 4 or 8 for stricter detection.

    Returns
    -------
    dict[int, dict[int, int]]
        Nested dictionary: {file_idx: {traj_idx: first_steady_timestep}}
        Each entry indicates the first timestep of a run of min_consecutive
        consecutive steady state steps.
    """
    logging.info(
        f"Computing steady state indices with {num_workers} workers, "
        f"atol={atol}, min_consecutive={min_consecutive}, dataset size={len(dataset)}"
    )

    # Prepare arguments for parallel processing
    args_list = [(dataset, traj_idx, atol, min_consecutive) for traj_idx in range(len(dataset))]

    # Use multiprocessing pool to check indices in parallel
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(_check_steady_state_index, args_list)

    # Build per-file, per-trajectory structure
    # Need to map global traj_idx back to (file_idx, local_traj_idx)
    steady_state_info: dict[int, dict[int, int]] = {}

    # Build cumulative trajectory counts per file
    traj_offsets = [0]
    for n_traj in dataset.n_trajectories_per_file:
        traj_offsets.append(traj_offsets[-1] + n_traj)

    for res in results:
        if res is not None:
            global_traj_idx, first_steady_t = res
            # Find which file this trajectory belongs to
            file_idx = int(
                np.searchsorted(traj_offsets[1:], global_traj_idx, side="right")
            )
            local_traj_idx = global_traj_idx - traj_offsets[file_idx]

            if file_idx not in steady_state_info:
                steady_state_info[file_idx] = {}
            steady_state_info[file_idx][local_traj_idx] = first_steady_t

    total_steady = sum(len(v) for v in steady_state_info.values())
    logging.info(
        f"Found {total_steady} trajectories with steady states "
        f"out of {len(dataset)} total trajectories"
    )

    return steady_state_info


def main():
    """
    Main function to precompute steady state indices for a dataset.

    This can be run as a standalone script to precompute indices for all splits.
    """
    parser = argparse.ArgumentParser(
        description="Precompute steady state indices for physics datasets"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the dataset (e.g., 'rayleigh_benard')",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-2,
        help="Absolute tolerance for steady state detection (default: 1e-2)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "valid", "test"],
        help="Dataset splits to process (default: train valid test)",
    )
    parser.add_argument(
        "--use_normalization",
        action="store_true",
        default=True,
        help="Use normalization when loading dataset",
    )
    parser.add_argument(
        "--min_consecutive",
        type=int,
        default=1,
        help="Minimum consecutive steady state steps required (default: 1)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Get data directory from environment
    data_dir = os.environ.get("DATA_DIR")
    if data_dir is None:
        raise ValueError("DATA_DIR environment variable not set")

    data_dir = Path(data_dir)
    dataset_base_path = data_dir / args.dataset_name

    # Path to save steady state info (per-trajectory format)
    indices_path = dataset_base_path / "steady_state_info.json"

    # Load existing JSON or create new structure
    if indices_path.exists():
        with open(indices_path, "r") as f:
            indices_data = json.load(f)
        logging.info(f"Loaded existing steady_state_info.json from {indices_path}")
    else:
        indices_data = {}
        logging.info(f"Creating new steady_state_info.json at {indices_path}")

    # Process each split
    for split in args.splits:
        logging.info(f"\n{'=' * 60}")
        logging.info(f"Processing split: {split}")
        logging.info(f"{'=' * 60}")

        # Check if already computed
        if split in indices_data and indices_data[split]:
            total_trajectories = sum(
                len(traj_dict) for traj_dict in indices_data[split].values()
            )
            logging.info(
                f"Steady state info for {split} already exists. "
                f"Found {total_trajectories} trajectories with steady states. Skipping."
            )
            continue

        # Build paths
        split_path = dataset_base_path / f"data/{split}"
        norm_path = dataset_base_path / "stats.yaml"

        if not split_path.exists():
            logging.warning(f"Split path {split_path} does not exist. Skipping.")
            continue

        # Create dataset
        logging.info(f"Loading dataset from {split_path}")
        dataset = _get_dataset(
            data_dir=split_path,
            norm_path=norm_path,
            n_steps_input=1,
            n_steps_output=1,
            use_normalization=args.use_normalization,
            normalization_type=ZScoreNormalization,
            dt_stride=1,
            full_trajectory_mode=True,
            max_rollout_steps=10000,
            restrict_indices=None,
        )

        logging.info(f"Dataset loaded with {len(dataset)} trajectories.")

        # Compute steady state info in parallel
        steady_state_info = compute_steadystate_indices(
            dataset=dataset,
            atol=args.atol,
            num_workers=args.num_workers,
            min_consecutive=args.min_consecutive,
        )

        # Convert int keys to strings for JSON compatibility
        # Format: {file_idx_str: {traj_idx_str: first_steady_timestep}}
        steady_state_info_json = {
            str(file_idx): {str(traj_idx): timestep for traj_idx, timestep in traj_dict.items()}
            for file_idx, traj_dict in steady_state_info.items()
        }

        indices_data[split] = steady_state_info_json

        # Write to file after each split (in case of interruption)
        with open(indices_path, "w") as f:
            json.dump(indices_data, f, indent=2)

        total_trajectories = sum(len(v) for v in steady_state_info.values())
        logging.info(
            f"Saved steady state info for {total_trajectories} trajectories "
            f"to {indices_path}"
        )

    logging.info(f"\n{'=' * 60}")
    logging.info("All splits processed successfully!")
    logging.info(f"Results saved to: {indices_path}")
    logging.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
