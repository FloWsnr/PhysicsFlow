from pathlib import Path
import pytest

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from physicsflow.train.eval import Evaluator
from physicsflow.models.flow_matching.flow_matching_model import FlowMatchingModel
from physicsflow.models.flow_matching.schedulers import CondOTScheduler


class SimpleVelocityNet(nn.Module):
    """Simple velocity network for testing."""

    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.tensor([0.1]))

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Return a simple transformation of input
        return x_t * self.param


@pytest.fixture
def model():
    velocity_net = SimpleVelocityNet()
    return FlowMatchingModel(velocity_net=velocity_net, scheduler=CondOTScheduler())


@pytest.fixture
def metrics():
    mse = torch.nn.MSELoss()
    mae = torch.nn.L1Loss()
    return {
        "mse": mse,
        "mae": mae,
    }


@pytest.fixture
def real_dataloader() -> DataLoader:
    """Create a real PyTorch DataLoader for testing."""
    # Create dummy data in the dict format expected by evaluator
    output_data = torch.randn(4, 10, 10)

    # Create dataset with proper dict format
    class TestDataset(torch.utils.data.Dataset):
        def __init__(self, outputs):
            self.outputs = outputs

        def __len__(self):
            return len(self.outputs)

        def __getitem__(self, idx):
            return {
                "output_fields": self.outputs[idx],
                "constant_scalars": torch.randn(3),
            }

    dataset = TestDataset(output_data)
    return DataLoader(dataset, batch_size=2, shuffle=False)


def test_eval(
    real_dataloader: DataLoader,
    model: torch.nn.Module,
    tmp_path: Path,
    metrics: dict[str, torch.nn.Module],
):
    evaluator = Evaluator(
        model=model,
        dataloader=real_dataloader,
        metrics=metrics,
        eval_dir=tmp_path,
    )
    eval_metrics = evaluator.eval()

    # user-provided metrics still work
    for metric_name in metrics:
        assert metric_name in eval_metrics
        assert torch.isfinite(eval_metrics[metric_name])

    # built-in flow-matching metrics are always included
    assert Evaluator.BUILTIN_REL_L2_KEY in eval_metrics
    assert Evaluator.BUILTIN_COSINE_ERROR_KEY in eval_metrics
    assert Evaluator.BUILTIN_MMD_KEY in eval_metrics
    assert torch.isfinite(eval_metrics[Evaluator.BUILTIN_REL_L2_KEY])
    assert torch.isfinite(eval_metrics[Evaluator.BUILTIN_COSINE_ERROR_KEY])
    assert torch.isfinite(eval_metrics[Evaluator.BUILTIN_MMD_KEY])

    # t-binned metrics are reported for each bin; bins without samples are NaN.
    rel_l2_bin_keys = [
        k for k in eval_metrics if k.startswith(f"{Evaluator.BUILTIN_REL_L2_KEY}_t")
    ]
    cosine_bin_keys = [
        k
        for k in eval_metrics
        if k.startswith(f"{Evaluator.BUILTIN_COSINE_ERROR_KEY}_t")
    ]
    assert len(rel_l2_bin_keys) == evaluator.time_bin_count
    assert len(cosine_bin_keys) == evaluator.time_bin_count
    assert any(torch.isfinite(eval_metrics[k]) for k in rel_l2_bin_keys)
    assert any(torch.isfinite(eval_metrics[k]) for k in cosine_bin_keys)


def test_eval_tiny_fraction_uses_at_least_one_batch(
    real_dataloader: DataLoader,
    model: torch.nn.Module,
    tmp_path: Path,
    metrics: dict[str, torch.nn.Module],
):
    evaluator = Evaluator(
        model=model,
        dataloader=real_dataloader,
        metrics=metrics,
        eval_dir=tmp_path,
        eval_fraction=0.1,
        time_bin_count=1,
    )
    eval_metrics = evaluator.eval()
    for metric_value in eval_metrics.values():
        assert torch.isfinite(metric_value)


def test_eval_fraction_zero_raises(
    real_dataloader: DataLoader,
    model: torch.nn.Module,
    tmp_path: Path,
    metrics: dict[str, torch.nn.Module],
):
    with pytest.raises(ValueError, match="eval_fraction must be in the range"):
        Evaluator(
            model=model,
            dataloader=real_dataloader,
            metrics=metrics,
            eval_dir=tmp_path,
            eval_fraction=0.0,
        )
