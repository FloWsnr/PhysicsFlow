"""Tests for DiT modulation/conditioning layers."""

import pytest
import torch

from physicsflow.models.backbone.dit.modulation import (
    AdaLNModulation,
    AdaLNBlock,
    CrossAttentionConditioning,
    FinalLayerModulation,
    modulate,
)


class TestModulateFunction:
    """Tests for modulate utility function."""

    def test_identity_with_zeros(self):
        """Test modulate is identity when shift=0, scale=0."""
        x = torch.randn(2, 100, 256)
        shift = torch.zeros(2, 256)
        scale = torch.zeros(2, 256)
        out = modulate(x, shift, scale)
        assert torch.allclose(out, x)

    def test_shift_only(self):
        """Test modulate with shift only."""
        x = torch.randn(2, 100, 256)
        shift = torch.ones(2, 256) * 0.5
        scale = torch.zeros(2, 256)
        out = modulate(x, shift, scale)
        expected = x + 0.5
        assert torch.allclose(out, expected)

    def test_scale_only(self):
        """Test modulate with scale only."""
        x = torch.randn(2, 100, 256)
        shift = torch.zeros(2, 256)
        scale = torch.ones(2, 256) * 0.5
        out = modulate(x, shift, scale)
        expected = x * 1.5
        assert torch.allclose(out, expected)

    def test_output_shape(self):
        """Test output shape matches input."""
        x = torch.randn(4, 50, 128)
        shift = torch.randn(4, 128)
        scale = torch.randn(4, 128)
        out = modulate(x, shift, scale)
        assert out.shape == x.shape


class TestAdaLNModulation:
    """Tests for AdaLNModulation."""

    def test_output_count(self):
        """Test AdaLN produces 6 parameters."""
        adaln = AdaLNModulation(hidden_dim=256, cond_embed_dim=128)
        c = torch.randn(2, 128)
        outputs = adaln(c)
        assert len(outputs) == 6

    def test_output_shapes(self):
        """Test AdaLN output shapes."""
        adaln = AdaLNModulation(hidden_dim=256, cond_embed_dim=128)
        c = torch.randn(4, 128)
        outputs = adaln(c)
        for out in outputs:
            assert out.shape == (4, 256)

    def test_zero_initialization(self):
        """Test that modulation starts near zero."""
        adaln = AdaLNModulation(256, 128)
        c = torch.randn(1, 128)
        with torch.no_grad():
            outputs = adaln(c)
        for out in outputs:
            # Should be close to zero initially (within numerical precision)
            assert out.abs().max() < 1e-5

    def test_gradient_flow(self):
        """Test gradients flow through AdaLN."""
        adaln = AdaLNModulation(256, 128)
        c = torch.randn(2, 128, requires_grad=True)
        outputs = adaln(c)
        loss = sum(out.sum() for out in outputs)
        loss.backward()
        assert c.grad is not None


class TestAdaLNBlock:
    """Tests for AdaLNBlock."""

    def test_output_shape(self):
        """Test output shape matches input."""
        block = AdaLNBlock(hidden_dim=256)
        x = torch.randn(2, 100, 256)
        shift = torch.randn(2, 256)
        scale = torch.randn(2, 256)
        out = block(x, shift, scale)
        assert out.shape == x.shape

    def test_normalization_applied(self):
        """Test that normalization is applied."""
        block = AdaLNBlock(hidden_dim=256)
        x = torch.randn(2, 100, 256) * 10 + 5  # Non-normalized input
        shift = torch.zeros(2, 256)
        scale = torch.zeros(2, 256)
        out = block(x, shift, scale)
        # After normalization, mean should be ~0, std ~1
        assert out.mean().abs() < 0.5
        assert (out.std() - 1.0).abs() < 0.5


class TestCrossAttentionConditioning:
    """Tests for CrossAttentionConditioning."""

    def test_output_shape(self):
        """Test cross-attention output shape."""
        cross_attn = CrossAttentionConditioning(
            hidden_dim=256, cond_embed_dim=128, num_heads=8
        )
        x = torch.randn(2, 100, 256)
        cond = torch.randn(2, 128)
        out = cross_attn(x, cond)
        assert out.shape == x.shape

    def test_output_shape_with_multiple_cond_tokens(self):
        """Test with multiple conditioning tokens."""
        cross_attn = CrossAttentionConditioning(256, 256, num_heads=8)
        x = torch.randn(2, 100, 256)
        cond = torch.randn(2, 4, 256)  # 4 conditioning tokens
        out = cross_attn(x, cond)
        assert out.shape == x.shape

    def test_gradient_flow(self):
        """Test gradients flow through cross-attention."""
        cross_attn = CrossAttentionConditioning(256, 128, num_heads=8)
        x = torch.randn(2, 100, 256, requires_grad=True)
        cond = torch.randn(2, 128, requires_grad=True)
        out = cross_attn(x, cond)
        out.sum().backward()
        assert x.grad is not None
        assert cond.grad is not None

    def test_different_embed_dims(self):
        """Test with different embedding dimensions."""
        for hidden_dim, cond_dim in [(256, 128), (512, 256), (128, 64)]:
            cross_attn = CrossAttentionConditioning(hidden_dim, cond_dim, num_heads=8)
            x = torch.randn(2, 50, hidden_dim)
            cond = torch.randn(2, cond_dim)
            out = cross_attn(x, cond)
            assert out.shape == x.shape


class TestFinalLayerModulation:
    """Tests for FinalLayerModulation."""

    def test_output_shape(self):
        """Test final layer output shape."""
        final = FinalLayerModulation(
            hidden_dim=256, out_dim=48, cond_embed_dim=128  # 3*4*4 = 48
        )
        x = torch.randn(2, 100, 256)
        c = torch.randn(2, 128)
        out = final(x, c)
        assert out.shape == (2, 100, 48)

    def test_zero_initialization(self):
        """Test that final layer is zero-initialized."""
        final = FinalLayerModulation(256, 48, 128)
        c = torch.randn(1, 128)
        x = torch.randn(1, 10, 256)
        with torch.no_grad():
            out = final(x, c)
        # Output should be near zero initially
        assert out.abs().max() < 1e-5

    def test_gradient_flow(self):
        """Test gradients flow through final layer."""
        final = FinalLayerModulation(256, 48, 128)
        x = torch.randn(2, 100, 256, requires_grad=True)
        c = torch.randn(2, 128, requires_grad=True)
        out = final(x, c)
        out.sum().backward()
        assert x.grad is not None
        assert c.grad is not None
