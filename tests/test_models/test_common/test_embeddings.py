"""Tests for embedding utilities."""

import pytest
import torch

from physicsflow.models.common import (
    sinusoidal_embedding,
    TimeEmbedding,
    ConditioningProjection,
)


class TestSinusoidalEmbedding:
    def test_output_shape(self):
        t = torch.tensor([0.0, 0.5, 1.0])
        dim = 64
        emb = sinusoidal_embedding(t, dim)
        assert emb.shape == (3, dim)

    def test_output_shape_2d_input(self):
        t = torch.tensor([[0.0], [0.5], [1.0]])
        dim = 64
        emb = sinusoidal_embedding(t, dim)
        assert emb.shape == (3, dim)

    def test_different_times_different_embeddings(self):
        t = torch.tensor([0.0, 0.5, 1.0])
        emb = sinusoidal_embedding(t, 64)
        # Each time should have unique embedding
        assert not torch.allclose(emb[0], emb[1])
        assert not torch.allclose(emb[1], emb[2])

    def test_deterministic(self):
        t = torch.tensor([0.25, 0.75])
        emb1 = sinusoidal_embedding(t, 32)
        emb2 = sinusoidal_embedding(t, 32)
        assert torch.allclose(emb1, emb2)


class TestTimeEmbedding:
    def test_output_shape(self):
        time_emb = TimeEmbedding(dim=64)
        t = torch.tensor([0.0, 0.5, 1.0])
        emb = time_emb(t)
        assert emb.shape == (3, 64)

    def test_learnable_parameters(self):
        time_emb = TimeEmbedding(dim=64)
        # Should have learnable MLP parameters
        params = list(time_emb.parameters())
        assert len(params) > 0

    def test_gradient_flow(self):
        time_emb = TimeEmbedding(dim=64)
        t = torch.tensor([0.5], requires_grad=False)
        emb = time_emb(t)
        loss = emb.sum()
        loss.backward()
        # Check gradients exist for MLP parameters
        for param in time_emb.parameters():
            assert param.grad is not None

    def test_odd_dim_raises(self):
        with pytest.raises(ValueError, match="dim must be even"):
            TimeEmbedding(dim=63)


class TestConditioningProjection:
    def test_output_shape(self):
        proj = ConditioningProjection(input_dim=3, output_dim=64)
        c = torch.randn(4, 3)
        out = proj(c)
        assert out.shape == (4, 64)

    def test_gradient_flow(self):
        proj = ConditioningProjection(input_dim=3, output_dim=64)
        c = torch.randn(4, 3)
        out = proj(c)
        loss = out.sum()
        loss.backward()
        for param in proj.parameters():
            assert param.grad is not None
