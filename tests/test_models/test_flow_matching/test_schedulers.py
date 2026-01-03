"""Tests for flow matching schedulers."""

import math

import pytest
import torch

from physicsflow.models.flow_matching import (
    CondOTScheduler,
    CosineScheduler,
    LinearVPScheduler,
    PolynomialScheduler,
    get_scheduler,
)


class TestCondOTScheduler:
    @pytest.fixture
    def scheduler(self):
        return CondOTScheduler()

    def test_boundary_t0(self, scheduler):
        """At t=0: alpha=0, sigma=1 (pure noise)."""
        out = scheduler(torch.tensor([0.0]))
        assert torch.allclose(out.alpha_t, torch.tensor([0.0]))
        assert torch.allclose(out.sigma_t, torch.tensor([1.0]))

    def test_boundary_t1(self, scheduler):
        """At t=1: alpha=1, sigma=0 (pure data)."""
        out = scheduler(torch.tensor([1.0]))
        assert torch.allclose(out.alpha_t, torch.tensor([1.0]))
        assert torch.allclose(out.sigma_t, torch.tensor([0.0]))

    def test_midpoint(self, scheduler):
        """At t=0.5: equal mix of noise and data."""
        out = scheduler(torch.tensor([0.5]))
        assert torch.allclose(out.alpha_t, torch.tensor([0.5]))
        assert torch.allclose(out.sigma_t, torch.tensor([0.5]))

    def test_derivatives(self, scheduler):
        """Derivatives should be constant for linear scheduler."""
        out = scheduler(torch.tensor([0.0, 0.5, 1.0]))
        assert torch.allclose(out.d_alpha_t, torch.ones(3))
        assert torch.allclose(out.d_sigma_t, -torch.ones(3))

    def test_sample_path(self, scheduler):
        """Test path interpolation."""
        x_0 = torch.randn(2, 3, 4, 8, 8)
        x_1 = torch.randn(2, 3, 4, 8, 8)
        t = torch.tensor([0.5, 0.5])

        x_t = scheduler.sample_path(x_0, x_1, t)
        expected = 0.5 * x_1 + 0.5 * x_0
        assert torch.allclose(x_t, expected)

    def test_target_velocity(self, scheduler):
        """Test velocity computation."""
        x_0 = torch.randn(2, 3, 4, 8, 8)
        x_1 = torch.randn(2, 3, 4, 8, 8)
        t = torch.tensor([0.5, 0.5])

        v = scheduler.target_velocity(x_0, x_1, t)
        # For CondOT: v = 1 * x_1 + (-1) * x_0 = x_1 - x_0
        expected = x_1 - x_0
        assert torch.allclose(v, expected)


class TestCosineScheduler:
    @pytest.fixture
    def scheduler(self):
        return CosineScheduler()

    def test_boundary_t0(self, scheduler):
        """At t=0: alpha=0, sigma=1."""
        out = scheduler(torch.tensor([0.0]))
        assert torch.allclose(out.alpha_t, torch.tensor([0.0]), atol=1e-6)
        assert torch.allclose(out.sigma_t, torch.tensor([1.0]), atol=1e-6)

    def test_boundary_t1(self, scheduler):
        """At t=1: alpha=1, sigma=0."""
        out = scheduler(torch.tensor([1.0]))
        assert torch.allclose(out.alpha_t, torch.tensor([1.0]), atol=1e-6)
        assert torch.allclose(out.sigma_t, torch.tensor([0.0]), atol=1e-6)

    def test_variance_preserving(self, scheduler):
        """Cosine scheduler preserves unit norm: alpha^2 + sigma^2 = 1."""
        t = torch.linspace(0, 1, 11)
        out = scheduler(t)
        norm_sq = out.alpha_t**2 + out.sigma_t**2
        assert torch.allclose(norm_sq, torch.ones_like(norm_sq), atol=1e-6)

    def test_smooth_derivatives_at_boundaries(self, scheduler):
        """Derivatives should be finite at boundaries."""
        out = scheduler(torch.tensor([0.0, 1.0]))
        assert torch.isfinite(out.d_alpha_t).all()
        assert torch.isfinite(out.d_sigma_t).all()


class TestLinearVPScheduler:
    @pytest.fixture
    def scheduler(self):
        return LinearVPScheduler()

    def test_boundary_t0(self, scheduler):
        """At t=0: alpha=0, sigma=1."""
        out = scheduler(torch.tensor([0.0]))
        assert torch.allclose(out.alpha_t, torch.tensor([0.0]))
        assert torch.allclose(out.sigma_t, torch.tensor([1.0]), atol=1e-3)

    def test_variance_preserving(self, scheduler):
        """LinearVP: alpha^2 + sigma^2 = 1 for all t."""
        t = torch.linspace(0.01, 0.99, 10)  # Avoid exact boundaries
        out = scheduler(t)
        norm_sq = out.alpha_t**2 + out.sigma_t**2
        assert torch.allclose(norm_sq, torch.ones_like(norm_sq), atol=1e-3)


class TestPolynomialScheduler:
    def test_n1_equals_cond_ot(self):
        """With n=1, polynomial should equal CondOT."""
        poly = PolynomialScheduler(n=1.0)
        cond_ot = CondOTScheduler()

        t = torch.linspace(0.1, 0.9, 9)
        out_poly = poly(t)
        out_cond = cond_ot(t)

        assert torch.allclose(out_poly.alpha_t, out_cond.alpha_t, atol=1e-5)
        assert torch.allclose(out_poly.sigma_t, out_cond.sigma_t, atol=1e-5)

    def test_boundary_conditions(self):
        """Boundaries should be same regardless of n."""
        for n in [0.5, 1.0, 2.0, 3.0]:
            scheduler = PolynomialScheduler(n=n)

            out_0 = scheduler(torch.tensor([0.0]))
            assert torch.allclose(out_0.alpha_t, torch.tensor([0.0]), atol=1e-5)
            assert torch.allclose(out_0.sigma_t, torch.tensor([1.0]), atol=1e-5)

            out_1 = scheduler(torch.tensor([1.0]))
            assert torch.allclose(out_1.alpha_t, torch.tensor([1.0]), atol=1e-5)
            assert torch.allclose(out_1.sigma_t, torch.tensor([0.0]), atol=1e-5)


class TestGetScheduler:
    def test_factory_cond_ot(self):
        scheduler = get_scheduler("cond_ot")
        assert isinstance(scheduler, CondOTScheduler)

    def test_factory_cosine(self):
        scheduler = get_scheduler("cosine")
        assert isinstance(scheduler, CosineScheduler)

    def test_factory_linear_vp(self):
        scheduler = get_scheduler("linear_vp")
        assert isinstance(scheduler, LinearVPScheduler)

    def test_factory_polynomial_with_kwargs(self):
        scheduler = get_scheduler("polynomial", n=3.0)
        assert isinstance(scheduler, PolynomialScheduler)
        assert scheduler.n == 3.0

    def test_unknown_scheduler_raises(self):
        with pytest.raises(ValueError, match="Unknown scheduler"):
            get_scheduler("unknown_scheduler")
