import pytest
from pathlib import Path
from typing import Callable, Optional

import yaml
import h5py
import numpy as np


@pytest.fixture(scope="session")
def write_dummy_data() -> Callable[[Path, Optional[np.ndarray]], None]:
    """Create a factory function that generates dummy data following the Well formatting for testing purposes.

    Returns
    -------
    Callable[[Path, Optional[np.ndarray]], None]
        A function that takes a Path and writes dummy data to that location.
    """

    def _write_dummy_data(
        filename: Path,
        t0_data: Optional[np.ndarray] = None,
        t1_data: Optional[np.ndarray] = None,
    ) -> None:
        # Create dummy data
        param_a = 0.25
        param_b = 0.75
        dataset_name = "dummy_dataset"
        grid_type = "cartesian"
        n_spatial_dims = 2

        # raise error if only one of t0_data or t1_data is provided
        if (t0_data is None) != (t1_data is None):
            raise ValueError(
                "Both t0_data and t1_data must be provided together or both be None."
            )

        if t0_data is None:
            # T0 field: (n_trajectories, dim_t, dim_x, dim_y)
            t0_data = np.random.rand(2, 10, 32, 32).astype(np.float32)

        if t1_data is None:
            # T1 field: (n_trajectories, dim_t, dim_x, dim_y, channels)
            t1_data = np.random.rand(2, 10, 32, 32, 2).astype(np.float32)

        t0_constant = t0_data[:, 0, ...]

        n_trajectories = t1_data.shape[0]
        dim_t = t1_data.shape[1]
        dim_x = t1_data.shape[2]
        dim_y = t1_data.shape[3]

        x = np.linspace(0, 1, dim_x, dtype=np.float32)
        y = np.linspace(0, 1, dim_y, dtype=np.float32)
        t = np.linspace(0, 1, dim_t, dtype=np.float32)
        x_peridocity_mask = np.zeros_like(x).astype(bool)
        x_peridocity_mask[0] = x_peridocity_mask[-1]
        y_peridocity_mask = np.zeros_like(y).astype(bool)
        y_peridocity_mask[0] = y_peridocity_mask[-1]

        time_varying_scalar_values = np.random.rand(dim_t)

        # Write the data in the HDF5 file
        filename.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(filename, "w") as file:
            # Attributes
            file.attrs["a"] = param_a
            file.attrs["b"] = param_b
            file.attrs["dataset_name"] = dataset_name
            file.attrs["grid_type"] = grid_type
            file.attrs["n_spatial_dims"] = n_spatial_dims
            file.attrs["n_trajectories"] = n_trajectories
            file.attrs["simulation_parameters"] = ["a", "b"]
            # Boundary Conditions
            group = file.create_group("boundary_conditions")
            for key, val in zip(
                ["x_periodic", "y_periodic"], [x_peridocity_mask, y_peridocity_mask]
            ):
                sub_group = group.create_group(key)
                sub_group.attrs["associated_dims"] = key[0]
                sub_group.attrs["associated_fields"] = []
                sub_group.attrs["bc_type"] = "PERIODIC"
                sub_group.attrs["sample_varying"] = False
                sub_group.attrs["time_varying"] = False
                sub_group.create_dataset("mask", data=val)
            # Dimensions
            group = file.create_group("dimensions")
            group.attrs["spatial_dims"] = ["x", "y"]
            for key, val in zip(["time", "x", "y"], [t, x, y]):
                group.create_dataset(key, data=val)
                group[key].attrs["sample_varying"] = False
            # Scalars
            group = file.create_group("scalars")
            group.attrs["field_names"] = ["a", "b", "time_varying_scalar"]
            for key, val in zip(["a", "b"], [param_a, param_b]):
                group.create_dataset(key, data=np.array(val))
                group[key].attrs["time_varying"] = False
                group[key].attrs["sample_varying"] = False
            ## Time varying
            dset = group.create_dataset(
                "time_varying_scalar", data=time_varying_scalar_values
            )
            dset.attrs["time_varying"] = True
            dset.attrs["sample_varying"] = False

            # Fields
            ############### T0 Fields ###############
            group = file.create_group("t0_fields")
            group.attrs["field_names"] = [
                "constant_field",
                "variable_field1",
                "variable_field2",
            ]
            # Add a constant field regarding time
            dset = group.create_dataset("constant_field", data=t0_constant)
            dset.attrs["dim_varying"] = [True, True]
            dset.attrs["sample_varying"] = True
            dset.attrs["time_varying"] = False

            dset = group.create_dataset("variable_field1", data=t0_data)
            dset.attrs["dim_varying"] = [True, True]
            dset.attrs["sample_varying"] = True
            dset.attrs["time_varying"] = True

            dset = group.create_dataset("variable_field2", data=t0_data)
            dset.attrs["dim_varying"] = [True, True]
            dset.attrs["sample_varying"] = True
            dset.attrs["time_varying"] = True

            ############### T1 Fields ###############
            # Add a field varying both in time and space
            group = file.create_group("t1_fields")
            group.attrs["field_names"] = ["field1", "field2"]
            dset = group.create_dataset("field1", data=t1_data)
            dset.attrs["dim_varying"] = [True, True]
            dset.attrs["sample_varying"] = True
            dset.attrs["time_varying"] = True

            dset = group.create_dataset("field2", data=t1_data)
            dset.attrs["dim_varying"] = [True, True]
            dset.attrs["sample_varying"] = True
            dset.attrs["time_varying"] = True

            ############# T2 Fields ###############
            group = file.create_group("t2_fields")
            group.attrs["field_names"] = []

    return _write_dummy_data


@pytest.fixture(scope="session")
def write_dummy_dataset(
    tmp_path_factory: pytest.TempPathFactory,
    write_dummy_data: Callable[[Path, Optional[np.ndarray]], None],
) -> Callable:
    """Create a factory function that generates complete dummy datasets for testing.

    Returns
    -------
    Callable
        A function that takes dataset configuration and returns the path to the created dataset directory.
    """

    def _write_dummy_dataset(
        sub_path: str = "data",
        n_datasets: int = 2,
        t0_data: Optional[np.ndarray] = None,
        t1_data: Optional[np.ndarray] = None,
    ) -> Path:
        """Create a complete dummy dataset directory with HDF5 files and stats.yaml.

        Parameters
        ----------
        sub_path : str
            Subdirectory name for the dataset
        n_datasets : int
            Number of dataset files to create
        t0_data : Optional[np.ndarray]
            T0 field data to use in the datasets
        t1_data : Optional[np.ndarray]
            T1 field data to use in the datasets

        Returns
        -------
        Path
            Path to the directory containing the dummy data.
        """
        data_dir = tmp_path_factory.mktemp("dummy_data")
        data_dir = data_dir / sub_path
        data_dir.mkdir(parents=True, exist_ok=True)

        # Create dataset files
        for i in range(1, n_datasets + 1):
            write_dummy_data(
                filename=data_dir / f"dummy_dataset{i}.hdf5",
                t0_data=t0_data,
                t1_data=t1_data,
            )

        # Create stats.yaml file for normalization
        stats = {
            "mean": {
                "field1": 0.5,
                "field2": 0.5,
                "constant_field": 0.5,
                "variable_field1": 0.5,
                "variable_field2": 0.5,
            },
            "mean_delta": {
                "field1": 0.0,
                "field2": 0.0,
                "constant_field": 0.0,
                "variable_field1": 0.0,
                "variable_field2": 0.0,
            },
            "rms": {
                "field1": 0.6,
                "field2": 0.6,
                "constant_field": 0.6,
                "variable_field1": 0.6,
                "variable_field2": 0.6,
            },
            "rms_delta": {
                "field1": 0.1,
                "field2": 0.1,
                "constant_field": 0.1,
                "variable_field1": 0.1,
                "variable_field2": 0.1,
            },
            "std": {
                "field1": 0.28867513,
                "field2": 0.28867513,
                "constant_field": 0.28867513,
                "variable_field1": 0.28867513,
                "variable_field2": 0.28867513,
            },
            "std_delta": {
                "field1": 0.1,
                "field2": 0.1,
                "constant_field": 0.1,
                "variable_field1": 0.1,
                "variable_field2": 0.1,
            },
        }

        # Write stats.yaml in the data directory
        stats_file = data_dir / "stats.yaml"
        with open(stats_file, "w") as f:
            yaml.dump(stats, f)

        return data_dir

    return _write_dummy_dataset
