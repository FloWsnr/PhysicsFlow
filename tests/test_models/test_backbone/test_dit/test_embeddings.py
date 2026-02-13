"""Tests for DiT embedding layers."""

import pytest
import torch

from physicsflow.models.dit.embeddings import (
    PatchEmbed3D,
    SpatioTemporalPosEmbed,
)


class TestPatchEmbed3D:
    """Tests for PatchEmbed3D."""

    def test_output_shape(self):
        """Test correct output dimensions."""
        embed = PatchEmbed3D(in_channels=3, hidden_dim=64, patch_size=(2, 2))
        x = torch.randn(2, 3, 4, 16, 16)
        out = embed(x)
        # Expected: (2, 4*8*8, 64) = (2, 256, 64)
        assert out.shape == (2, 256, 64)

    def test_different_patch_sizes(self):
        """Test various patch sizes."""
        for patch_size in [(2, 2), (4, 4)]:
            embed = PatchEmbed3D(3, 64, patch_size)
            x = torch.randn(1, 3, 4, 16, 16)
            out = embed(x)
            num_patches_h = 16 // patch_size[0]
            num_patches_w = 16 // patch_size[1]
            num_patches = 4 * num_patches_h * num_patches_w
            assert out.shape == (1, num_patches, 64)

    def test_different_input_channels(self):
        """Test different input channel counts."""
        for in_channels in [1, 5]:
            embed = PatchEmbed3D(in_channels, 64, (2, 2))
            x = torch.randn(2, in_channels, 4, 8, 8)
            out = embed(x)
            assert out.shape == (2, 4 * 4 * 4, 64)

    def test_gradient_flow(self):
        """Test gradients flow through embedding."""
        embed = PatchEmbed3D(3, 64, (2, 2))
        x = torch.randn(2, 3, 4, 8, 8, requires_grad=True)
        out = embed(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_get_num_patches(self):
        """Test num patches calculation."""
        embed = PatchEmbed3D(3, 64, (4, 4))
        h, w, t = embed.get_num_patches((16, 16), 4)
        assert h == 4
        assert w == 4
        assert t == 4


class TestSpatioTemporalPosEmbed:
    """Tests for SpatioTemporalPosEmbed."""

    def test_output_shape_learnable(self):
        """Test positional embedding shape with learnable embeddings."""
        pos_embed = SpatioTemporalPosEmbed(
            hidden_dim=64,
            max_spatial_size=(8, 8),
            max_temporal_size=4,
            learnable=True,
        )
        out = pos_embed(4, 4, 2)  # 4x4 spatial, 2 frames
        assert out.shape == (1, 2 * 4 * 4, 64)

    def test_output_shape_sinusoidal(self):
        """Test positional embedding shape with sinusoidal embeddings."""
        pos_embed = SpatioTemporalPosEmbed(
            hidden_dim=64,
            max_spatial_size=(8, 8),
            max_temporal_size=4,
            learnable=False,
        )
        out = pos_embed(4, 4, 2)
        assert out.shape == (1, 2 * 4 * 4, 64)

    def test_learnable_parameters(self):
        """Test that learnable embeddings have parameters."""
        pos_embed = SpatioTemporalPosEmbed(64, (8, 8), 4, learnable=True)
        params = list(pos_embed.parameters())
        assert len(params) == 2  # spatial and temporal
        assert params[0].requires_grad
        assert params[1].requires_grad

    def test_sinusoidal_no_parameters(self):
        """Test that sinusoidal embeddings have no learnable parameters."""
        pos_embed = SpatioTemporalPosEmbed(64, (8, 8), 4, learnable=False)
        params = list(pos_embed.parameters())
        assert len(params) == 0

    def test_different_sizes(self):
        """Test different spatial and temporal sizes."""
        pos_embed = SpatioTemporalPosEmbed(64, (8, 8), 8, learnable=True)
        for h, w, t in [(4, 4, 2), (8, 8, 4), (8, 8, 8)]:
            out = pos_embed(h, w, t)
            assert out.shape == (1, t * h * w, 64)

    def test_interpolation(self):
        """Test positional embedding interpolation."""
        pos_embed = SpatioTemporalPosEmbed(64, (4, 4), 2, learnable=True)
        # Interpolate to larger size
        out = pos_embed.interpolate(8, 8, 4)
        assert out.shape == (1, 4 * 8 * 8, 64)

    def test_deterministic(self):
        """Test that embeddings are deterministic."""
        pos_embed = SpatioTemporalPosEmbed(64, (8, 8), 4, learnable=True)
        out1 = pos_embed(4, 4, 2)
        out2 = pos_embed(4, 4, 2)
        assert torch.allclose(out1, out2)
