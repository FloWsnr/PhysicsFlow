"""Tests for the model factory (get_model)."""

import pytest
import torch

from physicsflow.models.model_utils import get_model
from physicsflow.models.flow_matching import FlowMatchingModel
from physicsflow.models.dit import DiTBackbone


@pytest.fixture
def model_config() -> dict:
    """Minimal config dict for get_model."""
    return {
        "size": "S",
        "in_channels": 2,
        "spatial_size": [16, 16],
        "temporal_size": 4,
        "cond_dim": 3,
        "scheduler": {
            "type": "cond_ot",
            "params": {},
        },
    }


class TestGetModel:
    def test_returns_flow_matching_model(self, model_config: dict):
        model = get_model(model_config)
        assert isinstance(model, FlowMatchingModel)

    def test_velocity_net_is_dit(self, model_config: dict):
        model = get_model(model_config)
        assert isinstance(model.velocity_net, DiTBackbone)

    def test_forward_pass(self, model_config: dict):
        model = get_model(model_config)
        B, C, T, H, W = 2, 2, 4, 16, 16
        x_1 = torch.randn(B, C, T, H, W)
        cond = torch.randn(B, 3)
        output = model(x_1, cond)
        assert output.predicted_velocity.shape == (B, C, T, H, W)

    @pytest.mark.parametrize("scheduler_type", ["cond_ot", "cosine", "linear_vp", "polynomial"])
    def test_different_schedulers(self, model_config: dict, scheduler_type: str):
        model_config["scheduler"] = {
            "type": scheduler_type,
            "params": {"n": 2} if scheduler_type == "polynomial" else {},
        }
        model = get_model(model_config)
        assert isinstance(model, FlowMatchingModel)

    def test_different_sizes(self, model_config: dict):
        model_config["size"] = "B"
        model = get_model(model_config)
        assert model.velocity_net.hidden_dim == 768
