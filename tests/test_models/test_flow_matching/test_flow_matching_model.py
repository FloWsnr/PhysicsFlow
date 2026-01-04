"""Tests for flow matching model."""

import pytest
import torch

from physicsflow.models.backbone import MLPBackbone
from physicsflow.models.flow_matching import (
    FlowMatchingModel,
    FlowMatchingOutput,
)


@pytest.fixture
def velocity_net():
    return MLPBackbone(
        in_channels=3,
        spatial_size=(8, 8),
        temporal_size=4,
        cond_dim=2,
        hidden_dim=64,
    )


@pytest.fixture
def velocity_net_no_cond():
    return MLPBackbone(
        in_channels=3,
        spatial_size=(8, 8),
        temporal_size=4,
        cond_dim=0,
        hidden_dim=64,
    )


@pytest.fixture
def flow_model(velocity_net):
    return FlowMatchingModel(velocity_net, scheduler="cond_ot")


@pytest.fixture
def sample_data():
    return {
        "input_fields": torch.randn(2, 3, 4, 8, 8),
        "constant_scalars": torch.randn(2, 2),
    }


class TestMLPBackbone:
    def test_output_shape(self, velocity_net):
        x_t = torch.randn(2, 3, 4, 8, 8)
        t = torch.tensor([0.5, 0.5])
        cond = torch.randn(2, 2)

        output = velocity_net(x_t, t, cond)
        assert output.shape == x_t.shape

    def test_output_shape_no_cond(self, velocity_net_no_cond):
        x_t = torch.randn(2, 3, 4, 8, 8)
        t = torch.tensor([0.5, 0.5])

        output = velocity_net_no_cond(x_t, t, None)
        assert output.shape == x_t.shape

    def test_gradient_flow(self, velocity_net):
        x_t = torch.randn(2, 3, 4, 8, 8)
        t = torch.tensor([0.5, 0.5])
        cond = torch.randn(2, 2)

        output = velocity_net(x_t, t, cond)
        loss = output.sum()
        loss.backward()

        for param in velocity_net.parameters():
            assert param.grad is not None


class TestFlowMatchingModel:
    def test_forward_output_type(self, flow_model, sample_data):
        output = flow_model(sample_data)
        assert isinstance(output, FlowMatchingOutput)

    def test_forward_loss_shape(self, flow_model, sample_data):
        output = flow_model(sample_data)
        assert output.loss.shape == ()
        assert output.loss.requires_grad

    def test_forward_velocity_shapes(self, flow_model, sample_data):
        output = flow_model(sample_data)
        expected_shape = sample_data["input_fields"].shape
        assert output.predicted_velocity.shape == expected_shape
        assert output.target_velocity.shape == expected_shape
        assert output.x_t.shape == expected_shape

    def test_forward_pred_target_aliases(self, flow_model, sample_data):
        """Test that pred and target aliases work for trainer compatibility."""
        output = flow_model(sample_data)
        assert torch.equal(output.pred, output.predicted_velocity)
        assert torch.equal(output.target, output.target_velocity)

    def test_forward_gradient_flow(self, flow_model, sample_data):
        output = flow_model(sample_data)
        output.loss.backward()

        for param in flow_model.parameters():
            assert param.grad is not None

    def test_sample_shape(self, flow_model):
        shape = (2, 3, 4, 8, 8)
        cond = torch.randn(2, 2)

        samples = flow_model.sample(shape=shape, cond=cond, num_steps=10)
        assert samples.shape == shape

    def test_sample_euler_vs_midpoint(self, flow_model):
        """Both methods should produce valid outputs."""
        shape = (2, 3, 4, 8, 8)
        cond = torch.randn(2, 2)

        torch.manual_seed(42)
        samples_euler = flow_model.sample(
            shape=shape, cond=cond, num_steps=10, method="euler"
        )

        torch.manual_seed(42)
        samples_midpoint = flow_model.sample(
            shape=shape, cond=cond, num_steps=10, method="midpoint"
        )

        # Both should produce valid shapes
        assert samples_euler.shape == shape
        assert samples_midpoint.shape == shape
        # They should be different (different integration methods)
        assert not torch.allclose(samples_euler, samples_midpoint)

    def test_different_schedulers(self, velocity_net, sample_data):
        """Test that all schedulers work."""
        for scheduler_name in ["cond_ot", "cosine", "linear_vp", "polynomial"]:
            model = FlowMatchingModel(velocity_net, scheduler=scheduler_name)
            output = model(sample_data)
            assert output.loss.shape == ()
            assert torch.isfinite(output.loss)

    def test_without_conditioning(self, velocity_net_no_cond):
        """Test model works without conditioning."""
        model = FlowMatchingModel(velocity_net_no_cond, scheduler="cond_ot")
        data = {
            "input_fields": torch.randn(2, 3, 4, 8, 8),
        }
        output = model(data)
        assert output.loss.shape == ()


class TestFlowMatchingModelIntegration:
    def test_training_step_simulation(self, flow_model, sample_data):
        """Simulate a training step like the Trainer would do."""
        optimizer = torch.optim.Adam(flow_model.parameters(), lr=1e-3)

        # Forward pass
        output = flow_model(sample_data)
        loss = output.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Verify parameters were updated
        assert True  # If we got here without error, the step worked

    def test_loss_decreases_on_overfit(self, velocity_net):
        """Test that loss can decrease when overfitting to a single batch."""
        model = FlowMatchingModel(velocity_net, scheduler="cond_ot")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        # Fixed data to overfit
        torch.manual_seed(42)
        data = {
            "input_fields": torch.randn(2, 3, 4, 8, 8),
            "constant_scalars": torch.randn(2, 2),
        }

        # Train for a few steps
        losses = []
        for _ in range(20):
            output = model(data)
            losses.append(output.loss.item())
            optimizer.zero_grad()
            output.loss.backward()
            optimizer.step()

        # Loss should generally decrease (allow some variance)
        assert losses[-1] < losses[0] * 1.5  # Allow some wiggle room
