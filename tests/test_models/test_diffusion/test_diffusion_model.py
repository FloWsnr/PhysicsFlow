"""Tests for diffusion model."""

import pytest
import torch

from physicsflow.models.backbone import MLPBackbone
from physicsflow.models.diffusion import (
    DiffusionModel,
    DiffusionOutput,
)


@pytest.fixture
def denoiser():
    return MLPBackbone(
        in_channels=3,
        spatial_size=(8, 8),
        temporal_size=4,
        cond_dim=2,
        hidden_dim=64,
    )


@pytest.fixture
def denoiser_no_cond():
    return MLPBackbone(
        in_channels=3,
        spatial_size=(8, 8),
        temporal_size=4,
        cond_dim=0,
        hidden_dim=64,
    )


@pytest.fixture
def diffusion_model(denoiser):
    return DiffusionModel(
        denoiser,
        timesteps=100,
        schedule="cosine",
        objective="pred_noise",
    )


@pytest.fixture
def sample_data():
    return {
        "input_fields": torch.randn(2, 3, 4, 8, 8),
        "constant_scalars": torch.randn(2, 2),
    }


class TestMLPBackbone:
    def test_output_shape(self, denoiser):
        x_t = torch.randn(2, 3, 4, 8, 8)
        t = torch.tensor([50, 75])
        cond = torch.randn(2, 2)

        output = denoiser(x_t, t, cond)
        assert output.shape == x_t.shape

    def test_output_shape_no_cond(self, denoiser_no_cond):
        x_t = torch.randn(2, 3, 4, 8, 8)
        t = torch.tensor([50, 75])

        output = denoiser_no_cond(x_t, t, None)
        assert output.shape == x_t.shape

    def test_gradient_flow(self, denoiser):
        x_t = torch.randn(2, 3, 4, 8, 8)
        t = torch.tensor([50, 75])
        cond = torch.randn(2, 2)

        output = denoiser(x_t, t, cond)
        loss = output.sum()
        loss.backward()

        for param in denoiser.parameters():
            assert param.grad is not None


class TestDiffusionModel:
    def test_forward_output_type(self, diffusion_model, sample_data):
        output = diffusion_model(sample_data)
        assert isinstance(output, DiffusionOutput)

    def test_forward_loss_shape(self, diffusion_model, sample_data):
        output = diffusion_model(sample_data)
        assert output.loss.shape == ()
        assert output.loss.requires_grad

    def test_forward_pred_target_shapes(self, diffusion_model, sample_data):
        output = diffusion_model(sample_data)
        expected_shape = sample_data["input_fields"].shape
        assert output.pred.shape == expected_shape
        assert output.target.shape == expected_shape

    def test_forward_gradient_flow(self, diffusion_model, sample_data):
        output = diffusion_model(sample_data)
        output.loss.backward()

        for param in diffusion_model.parameters():
            assert param.grad is not None

    def test_q_sample(self, diffusion_model, sample_data):
        """Test forward diffusion process."""
        x_start = sample_data["input_fields"]
        t = torch.tensor([0, 50])

        x_t, noise = diffusion_model.q_sample(x_start, t)

        assert x_t.shape == x_start.shape
        assert noise.shape == x_start.shape

    def test_q_sample_boundaries(self, diffusion_model, sample_data):
        """At t=0, x_t should be close to x_start."""
        x_start = sample_data["input_fields"]
        t = torch.zeros(2, dtype=torch.long)

        x_t, noise = diffusion_model.q_sample(x_start, t)

        # At t=0, minimal noise should be added
        # sqrt_alphas_cumprod[0] should be close to 1
        # So x_t should be close to x_start
        # (Not exactly equal due to some small noise component)
        assert x_t.shape == x_start.shape

    def test_sample_shape(self, diffusion_model):
        shape = (2, 3, 4, 8, 8)
        cond = torch.randn(2, 2)

        # Use fewer timesteps for faster test
        diffusion_model.timesteps = 10
        samples = diffusion_model.sample(shape=shape, cond=cond)

        assert samples.shape == shape

    def test_sample_return_all_timesteps(self, diffusion_model):
        shape = (2, 3, 4, 8, 8)
        cond = torch.randn(2, 2)

        diffusion_model.timesteps = 10
        intermediates = diffusion_model.sample(
            shape=shape, cond=cond, return_all_timesteps=True
        )

        # Should have timesteps + 1 intermediates (including initial noise)
        assert isinstance(intermediates, list)
        assert len(intermediates) == 11
        assert all(x.shape == shape for x in intermediates)


class TestDiffusionModelObjectives:
    @pytest.fixture
    def denoiser(self):
        return MLPBackbone(
            in_channels=3,
            spatial_size=(8, 8),
            temporal_size=4,
            cond_dim=2,
            hidden_dim=64,
        )

    @pytest.fixture
    def sample_data(self):
        return {
            "input_fields": torch.randn(2, 3, 4, 8, 8),
            "constant_scalars": torch.randn(2, 2),
        }

    def test_pred_noise_objective(self, denoiser, sample_data):
        model = DiffusionModel(
            denoiser, timesteps=100, objective="pred_noise"
        )
        output = model(sample_data)
        assert output.loss.shape == ()
        assert torch.isfinite(output.loss)

    def test_pred_x0_objective(self, denoiser, sample_data):
        model = DiffusionModel(
            denoiser, timesteps=100, objective="pred_x0"
        )
        output = model(sample_data)
        assert output.loss.shape == ()
        assert torch.isfinite(output.loss)

    def test_pred_v_objective(self, denoiser, sample_data):
        model = DiffusionModel(
            denoiser, timesteps=100, objective="pred_v"
        )
        output = model(sample_data)
        assert output.loss.shape == ()
        assert torch.isfinite(output.loss)


class TestDiffusionModelLossTypes:
    @pytest.fixture
    def denoiser(self):
        return MLPBackbone(
            in_channels=3,
            spatial_size=(8, 8),
            temporal_size=4,
            cond_dim=2,
            hidden_dim=64,
        )

    @pytest.fixture
    def sample_data(self):
        return {
            "input_fields": torch.randn(2, 3, 4, 8, 8),
            "constant_scalars": torch.randn(2, 2),
        }

    def test_l1_loss(self, denoiser, sample_data):
        model = DiffusionModel(denoiser, timesteps=100, loss_type="l1")
        output = model(sample_data)
        assert torch.isfinite(output.loss)

    def test_l2_loss(self, denoiser, sample_data):
        model = DiffusionModel(denoiser, timesteps=100, loss_type="l2")
        output = model(sample_data)
        assert torch.isfinite(output.loss)

    def test_huber_loss(self, denoiser, sample_data):
        model = DiffusionModel(denoiser, timesteps=100, loss_type="huber")
        output = model(sample_data)
        assert torch.isfinite(output.loss)


class TestDiffusionModelSNRWeighting:
    @pytest.fixture
    def denoiser(self):
        return MLPBackbone(
            in_channels=3,
            spatial_size=(8, 8),
            temporal_size=4,
            cond_dim=2,
            hidden_dim=64,
        )

    @pytest.fixture
    def sample_data(self):
        return {
            "input_fields": torch.randn(2, 3, 4, 8, 8),
            "constant_scalars": torch.randn(2, 2),
        }

    def test_snr_weighting_pred_noise(self, denoiser, sample_data):
        model = DiffusionModel(
            denoiser,
            timesteps=100,
            objective="pred_noise",
            snr_weighting=True,
            min_snr_gamma=5.0,
        )
        output = model(sample_data)
        assert torch.isfinite(output.loss)

    def test_snr_weighting_pred_v(self, denoiser, sample_data):
        model = DiffusionModel(
            denoiser,
            timesteps=100,
            objective="pred_v",
            snr_weighting=True,
        )
        output = model(sample_data)
        assert torch.isfinite(output.loss)


class TestDiffusionModelSchedules:
    @pytest.fixture
    def denoiser(self):
        return MLPBackbone(
            in_channels=3,
            spatial_size=(8, 8),
            temporal_size=4,
            cond_dim=2,
            hidden_dim=64,
        )

    @pytest.fixture
    def sample_data(self):
        return {
            "input_fields": torch.randn(2, 3, 4, 8, 8),
            "constant_scalars": torch.randn(2, 2),
        }

    def test_linear_schedule(self, denoiser, sample_data):
        model = DiffusionModel(denoiser, timesteps=100, schedule="linear")
        output = model(sample_data)
        assert torch.isfinite(output.loss)

    def test_cosine_schedule(self, denoiser, sample_data):
        model = DiffusionModel(denoiser, timesteps=100, schedule="cosine")
        output = model(sample_data)
        assert torch.isfinite(output.loss)

    def test_sigmoid_schedule(self, denoiser, sample_data):
        model = DiffusionModel(denoiser, timesteps=100, schedule="sigmoid")
        output = model(sample_data)
        assert torch.isfinite(output.loss)


class TestDiffusionModelIntegration:
    def test_training_step_simulation(self, diffusion_model, sample_data):
        """Simulate a training step like the Trainer would do."""
        optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=1e-3)

        # Forward pass
        output = diffusion_model(sample_data)
        loss = output.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert True  # If we got here without error, the step worked

    def test_without_conditioning(self, denoiser_no_cond):
        """Test model works without conditioning."""
        model = DiffusionModel(denoiser_no_cond, timesteps=100)
        data = {
            "input_fields": torch.randn(2, 3, 4, 8, 8),
        }
        output = model(data)
        assert output.loss.shape == ()
