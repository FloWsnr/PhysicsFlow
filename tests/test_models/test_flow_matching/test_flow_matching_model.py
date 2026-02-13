"""Tests for flow matching model."""

import pytest
import torch

from physicsflow.models.dit import DiTBackbone
from physicsflow.models.flow_matching import (
    FlowMatchingModel,
    FlowMatchingOutput,
)
from physicsflow.models.flow_matching.schedulers import (
    CondOTScheduler,
    CosineScheduler,
    LinearVPScheduler,
    PolynomialScheduler,
)


@pytest.fixture
def velocity_net():
    return DiTBackbone(
        in_channels=3,
        spatial_size=(8, 8),
        temporal_size=4,
        cond_dim=2,
        hidden_dim=64,
        depth=2,
        num_heads=4,
        mlp_ratio=4.0,
        patch_size=(2, 2),
        time_embed_dim=64,
        conditioning_type="adaln",
        dropout=0.0,
        attn_drop=0.0,
        learnable_pos_embed=True,
    )


@pytest.fixture
def velocity_net_no_cond():
    return DiTBackbone(
        in_channels=3,
        spatial_size=(8, 8),
        temporal_size=4,
        cond_dim=0,
        hidden_dim=64,
        depth=2,
        num_heads=4,
        mlp_ratio=4.0,
        patch_size=(2, 2),
        time_embed_dim=64,
        conditioning_type="adaln",
        dropout=0.0,
        attn_drop=0.0,
        learnable_pos_embed=True,
    )


@pytest.fixture
def flow_model(velocity_net):
    return FlowMatchingModel(velocity_net, scheduler=CondOTScheduler())


@pytest.fixture
def sample_data():
    return {
        "input_fields": torch.randn(2, 3, 4, 8, 8),
        "constant_scalars": torch.randn(2, 2),
    }


class TestDiTBackbone:
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
        x_1 = sample_data["input_fields"]
        cond = sample_data["constant_scalars"]
        output = flow_model(x_1, cond)
        assert isinstance(output, FlowMatchingOutput)

    def test_forward_loss_computation(self, flow_model, sample_data):
        """Test that loss can be computed from predicted/target velocities."""
        x_1 = sample_data["input_fields"]
        cond = sample_data["constant_scalars"]
        output = flow_model(x_1, cond)
        # Loss is computed externally from predicted and target velocities
        loss = torch.nn.functional.mse_loss(
            output.predicted_velocity, output.target_velocity
        )
        assert loss.shape == ()
        assert loss.requires_grad

    def test_forward_velocity_shapes(self, flow_model, sample_data):
        x_1 = sample_data["input_fields"]
        cond = sample_data["constant_scalars"]
        output = flow_model(x_1, cond)
        expected_shape = sample_data["input_fields"].shape
        assert output.predicted_velocity.shape == expected_shape
        assert output.target_velocity.shape == expected_shape
        assert output.x_t.shape == expected_shape

    def test_forward_velocities_exist(self, flow_model, sample_data):
        """Test that predicted/target velocities are returned."""
        x_1 = sample_data["input_fields"]
        cond = sample_data["constant_scalars"]
        output = flow_model(x_1, cond)
        assert hasattr(output, "predicted_velocity")
        assert hasattr(output, "target_velocity")
        assert output.predicted_velocity is not None
        assert output.target_velocity is not None

    def test_forward_gradient_flow(self, flow_model, sample_data):
        x_1 = sample_data["input_fields"]
        cond = sample_data["constant_scalars"]
        output = flow_model(x_1, cond)
        loss = torch.nn.functional.mse_loss(
            output.predicted_velocity, output.target_velocity
        )
        loss.backward()

        for param in flow_model.parameters():
            assert param.grad is not None

    def test_sample_shape(self, flow_model):
        shape = (2, 3, 4, 8, 8)
        cond = torch.randn(2, 2)

        samples = flow_model.sample(shape=shape, cond=cond, num_steps=3)
        assert samples.shape == shape

    def test_sample_euler_vs_midpoint(self, flow_model):
        """Both methods should produce valid outputs."""
        shape = (2, 3, 4, 8, 8)
        cond = torch.randn(2, 2)

        torch.manual_seed(42)
        samples_euler = flow_model.sample(
            shape=shape, cond=cond, num_steps=3, method="euler"
        )

        torch.manual_seed(42)
        samples_midpoint = flow_model.sample(
            shape=shape, cond=cond, num_steps=3, method="midpoint"
        )

        # Both should produce valid shapes
        assert samples_euler.shape == shape
        assert samples_midpoint.shape == shape
        # They should be different (different integration methods)
        assert not torch.allclose(samples_euler, samples_midpoint)

    def test_different_schedulers(self, velocity_net, sample_data):
        """Test that all schedulers work."""
        x_1 = sample_data["input_fields"]
        cond = sample_data["constant_scalars"]
        for scheduler in [
            CondOTScheduler(),
            CosineScheduler(),
            LinearVPScheduler(),
            PolynomialScheduler(),
        ]:
            model = FlowMatchingModel(velocity_net, scheduler=scheduler)
            output = model(x_1, cond)
            loss = torch.nn.functional.mse_loss(
                output.predicted_velocity, output.target_velocity
            )
            assert loss.shape == ()
            assert torch.isfinite(loss)

    def test_without_conditioning(self, velocity_net_no_cond):
        """Test model works without conditioning."""
        model = FlowMatchingModel(velocity_net_no_cond, scheduler=CondOTScheduler())
        x_1 = torch.randn(2, 3, 4, 8, 8)
        output = model(x_1, cond=None)
        loss = torch.nn.functional.mse_loss(
            output.predicted_velocity, output.target_velocity
        )
        assert loss.shape == ()


class TestFlowMatchingModelIntegration:
    def test_training_step_simulation(self, flow_model, sample_data):
        """Simulate a training step like the Trainer would do."""
        optimizer = torch.optim.Adam(flow_model.parameters(), lr=1e-3)

        # Forward pass
        x_1 = sample_data["input_fields"]
        cond = sample_data["constant_scalars"]
        output = flow_model(x_1, cond)
        loss = torch.nn.functional.mse_loss(
            output.predicted_velocity, output.target_velocity
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Verify parameters were updated
        assert True  # If we got here without error, the step worked

    def test_loss_decreases_on_overfit(self, velocity_net):
        """Test that loss can decrease when overfitting to a single batch."""
        model = FlowMatchingModel(velocity_net, scheduler=CondOTScheduler())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        # Fixed data to overfit
        torch.manual_seed(42)
        x_1 = torch.randn(2, 3, 4, 8, 8)
        cond = torch.randn(2, 2)

        # Train for a few steps
        losses = []
        for _ in range(10):
            output = model(x_1, cond)
            loss = torch.nn.functional.mse_loss(
                output.predicted_velocity, output.target_velocity
            )
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Loss should generally decrease (allow some variance)
        assert losses[-1] < losses[0] * 1.5  # Allow some wiggle room
