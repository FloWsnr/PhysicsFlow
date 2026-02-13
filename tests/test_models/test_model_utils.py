"""Tests for the model factory (get_model)."""

import pytest
import torch

from physicsflow.models.model_utils import get_model
from physicsflow.models.flow_matching import FlowMatchingModel
from physicsflow.models.dit import DiTBackbone


def _make_config(**overrides) -> dict:
    """Minimal config dict for get_model."""
    config = {
        "size": "S",
        "in_channels": 2,
        "spatial_size": [8, 8],
        "temporal_size": 4,
        "cond_dim": 3,
        "scheduler": {
            "type": "cond_ot",
            "params": {},
        },
    }
    config.update(overrides)
    return config


class TestGetModel:
    @pytest.fixture(scope="class")
    def model(self):
        """Shared DiT-S model instance (expensive to construct)."""
        return get_model(_make_config())

    def test_returns_flow_matching_model(self, model):
        assert isinstance(model, FlowMatchingModel)

    def test_velocity_net_is_dit(self, model):
        assert isinstance(model.velocity_net, DiTBackbone)

    def test_forward_pass(self, model):
        B, C, T, H, W = 2, 2, 4, 8, 8
        x_1 = torch.randn(B, C, T, H, W)
        cond = torch.randn(B, 3)
        output = model(x_1, cond)
        assert output.predicted_velocity.shape == (B, C, T, H, W)

    @pytest.mark.parametrize("scheduler_type", ["cond_ot", "polynomial"])
    def test_different_schedulers(self, scheduler_type: str):
        config = _make_config(
            scheduler={
                "type": scheduler_type,
                "params": {"n": 2} if scheduler_type == "polynomial" else {},
            }
        )
        model = get_model(config)
        assert isinstance(model, FlowMatchingModel)

    def test_different_sizes(self):
        model = get_model(_make_config(size="B"))
        assert model.velocity_net.hidden_dim == 768
