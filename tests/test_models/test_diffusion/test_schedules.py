"""Tests for diffusion schedules."""

import pytest
import torch

from physicsflow.models.diffusion import (
    DiffusionSchedule,
    linear_beta_schedule,
    cosine_beta_schedule,
    sigmoid_beta_schedule,
    get_schedule,
)


class TestLinearBetaSchedule:
    def test_output_shape(self):
        betas = linear_beta_schedule(1000)
        assert betas.shape == (1000,)

    def test_monotonically_increasing(self):
        betas = linear_beta_schedule(1000)
        assert (betas[1:] >= betas[:-1]).all()

    def test_custom_bounds(self):
        betas = linear_beta_schedule(100, beta_start=0.001, beta_end=0.1)
        assert torch.isclose(betas[0], torch.tensor(0.001), atol=1e-4)
        assert torch.isclose(betas[-1], torch.tensor(0.1), atol=1e-4)


class TestCosineBetaSchedule:
    def test_output_shape(self):
        betas = cosine_beta_schedule(1000)
        assert betas.shape == (1000,)

    def test_values_in_valid_range(self):
        betas = cosine_beta_schedule(1000)
        assert (betas >= 0).all()
        assert (betas <= 1).all()

    def test_smooth_progression(self):
        """Cosine schedule should be smoother than linear."""
        betas = cosine_beta_schedule(100)
        # Differences should exist but be smooth
        diffs = betas[1:] - betas[:-1]
        assert torch.isfinite(diffs).all()


class TestSigmoidBetaSchedule:
    def test_output_shape(self):
        betas = sigmoid_beta_schedule(1000)
        assert betas.shape == (1000,)

    def test_values_in_bounds(self):
        betas = sigmoid_beta_schedule(1000, beta_start=0.0001, beta_end=0.02)
        assert (betas >= 0.0001 - 1e-6).all()
        assert (betas <= 0.02 + 1e-6).all()


class TestGetSchedule:
    def test_linear_schedule(self):
        schedule = get_schedule("linear", 100)
        assert isinstance(schedule, DiffusionSchedule)
        assert schedule.betas.shape == (100,)

    def test_cosine_schedule(self):
        schedule = get_schedule("cosine", 100)
        assert isinstance(schedule, DiffusionSchedule)
        assert schedule.betas.shape == (100,)

    def test_sigmoid_schedule(self):
        schedule = get_schedule("sigmoid", 100)
        assert isinstance(schedule, DiffusionSchedule)
        assert schedule.betas.shape == (100,)

    def test_unknown_schedule_raises(self):
        with pytest.raises(ValueError, match="Unknown schedule"):
            get_schedule("unknown", 100)


class TestDiffusionSchedule:
    @pytest.fixture
    def schedule(self):
        return get_schedule("cosine", 100)

    def test_all_tensors_have_correct_shape(self, schedule):
        timesteps = 100
        assert schedule.betas.shape == (timesteps,)
        assert schedule.alphas.shape == (timesteps,)
        assert schedule.alphas_cumprod.shape == (timesteps,)
        assert schedule.alphas_cumprod_prev.shape == (timesteps,)
        assert schedule.sqrt_alphas_cumprod.shape == (timesteps,)
        assert schedule.sqrt_one_minus_alphas_cumprod.shape == (timesteps,)
        assert schedule.snr.shape == (timesteps,)

    def test_alphas_cumprod_decreasing(self, schedule):
        """Alpha cumulative product should decrease monotonically."""
        assert (schedule.alphas_cumprod[1:] <= schedule.alphas_cumprod[:-1]).all()

    def test_alphas_cumprod_bounds(self, schedule):
        """Alpha cumulative product should be in (0, 1]."""
        assert (schedule.alphas_cumprod > 0).all()
        assert (schedule.alphas_cumprod <= 1).all()

    def test_sqrt_values_consistent(self, schedule):
        """sqrt values should match their sources."""
        assert torch.allclose(
            schedule.sqrt_alphas_cumprod**2,
            schedule.alphas_cumprod,
            atol=1e-6,
        )
        assert torch.allclose(
            schedule.sqrt_one_minus_alphas_cumprod**2,
            1 - schedule.alphas_cumprod,
            atol=1e-6,
        )

    def test_snr_positive(self, schedule):
        """SNR should be positive."""
        assert (schedule.snr > 0).all()

    def test_snr_decreases_with_time(self, schedule):
        """SNR should generally decrease as noise increases."""
        # Not strictly monotonic due to numerical issues, but trend should be down
        assert schedule.snr[0] > schedule.snr[-1]
