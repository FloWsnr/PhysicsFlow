from dataclasses import dataclass
from pathlib import Path
import pytest

import torch
from torch.utils.data import DataLoader

from physicsflow.train.eval import Evaluator


@dataclass
class SimpleOutput:
    """Simple output dataclass for testing."""

    loss: torch.Tensor
    pred: torch.Tensor
    target: torch.Tensor


class SimpleModel(torch.nn.Module):
    """Simple model that returns output dataclass."""

    def forward(self, data: dict):
        x = data["input_fields"]
        pred = x + 0.1  # Small modification
        target = x
        loss = torch.nn.functional.mse_loss(pred, target)
        return SimpleOutput(loss=loss, pred=pred, target=target)


@pytest.fixture
def model():
    return SimpleModel()


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
    input_data = torch.randn(4, 10, 10)

    # Create dataset with proper dict format
    class TestDataset(torch.utils.data.Dataset):
        def __init__(self, inputs):
            self.inputs = inputs

        def __len__(self):
            return len(self.inputs)

        def __getitem__(self, idx):
            return {
                "input_fields": self.inputs[idx],
                "constant_scalars": torch.randn(3),
            }

    dataset = TestDataset(input_data)
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
    losses = evaluator.eval()

    for metric_name, metric_value in losses.items():
        assert metric_value.item() != 0.0
