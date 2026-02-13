"""Tests for DiT transformer blocks."""

import pytest
import torch

from physicsflow.models.dit.blocks import (
    Attention,
    DiTBlock,
    FactorizedAttention,
    FinalLayer,
    Mlp,
)


class TestMlp:
    """Tests for MLP block."""

    def test_output_shape(self):
        """Test MLP output shape."""
        mlp = Mlp(in_features=64, hidden_features=128)
        x = torch.randn(2, 16, 64)
        out = mlp(x)
        assert out.shape == x.shape

    def test_custom_out_features(self):
        """Test MLP with different output dimension."""
        mlp = Mlp(in_features=64, hidden_features=128, out_features=32)
        x = torch.randn(2, 16, 64)
        out = mlp(x)
        assert out.shape == (2, 16, 32)

    def test_gradient_flow(self):
        """Test gradients flow through MLP."""
        mlp = Mlp(64, 128)
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = mlp(x)
        out.sum().backward()
        assert x.grad is not None


class TestAttention:
    """Tests for Attention block."""

    def test_output_shape(self):
        """Test attention output shape."""
        attn = Attention(dim=64, num_heads=4)
        x = torch.randn(2, 16, 64)
        out = attn(x)
        assert out.shape == x.shape

    def test_different_heads(self):
        """Test with different number of heads."""
        for num_heads in [4, 8]:
            attn = Attention(dim=64, num_heads=num_heads)
            x = torch.randn(2, 16, 64)
            out = attn(x)
            assert out.shape == x.shape

    def test_gradient_flow(self):
        """Test gradients flow through attention."""
        attn = Attention(64, 4)
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = attn(x)
        out.sum().backward()
        assert x.grad is not None


class TestFactorizedAttention:
    """Tests for FactorizedAttention."""

    def test_output_shape(self):
        """Test factorized attention preserves shape."""
        attn = FactorizedAttention(hidden_dim=64, num_heads=4)
        x = torch.randn(2, 20, 64)  # T=4, N=5
        out = attn(x, num_frames=4, num_patches=5)
        assert out.shape == x.shape

    def test_different_temporal_spatial(self):
        """Test different temporal/spatial combinations."""
        attn = FactorizedAttention(hidden_dim=64, num_heads=4)
        for t, n in [(4, 8), (8, 4)]:
            x = torch.randn(2, t * n, 64)
            out = attn(x, num_frames=t, num_patches=n)
            assert out.shape == x.shape

    def test_gradient_flow(self):
        """Test gradients flow through both attention paths."""
        attn = FactorizedAttention(64, 4)
        x = torch.randn(2, 20, 64, requires_grad=True)
        out = attn(x, num_frames=4, num_patches=5)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_batch_independence(self):
        """Test that batches are processed independently."""
        attn = FactorizedAttention(64, 4)
        x1 = torch.randn(1, 20, 64)
        x2 = torch.randn(1, 20, 64)
        x_batch = torch.cat([x1, x2], dim=0)

        out_batch = attn(x_batch, num_frames=4, num_patches=5)
        out1 = attn(x1, num_frames=4, num_patches=5)
        out2 = attn(x2, num_frames=4, num_patches=5)

        assert torch.allclose(out_batch[0], out1[0], atol=1e-5)
        assert torch.allclose(out_batch[1], out2[0], atol=1e-5)


class TestDiTBlock:
    """Tests for DiTBlock."""

    def test_adaln_forward(self):
        """Test DiT block with AdaLN conditioning."""
        block = DiTBlock(
            hidden_dim=64,
            num_heads=4,
            cond_embed_dim=64,
            conditioning_type="adaln",
        )
        x = torch.randn(2, 20, 64)
        c = torch.randn(2, 64)
        out = block(x, c, num_frames=4, num_patches=5)
        assert out.shape == x.shape

    def test_cross_attention_forward(self):
        """Test DiT block with cross-attention conditioning."""
        block = DiTBlock(
            hidden_dim=64,
            num_heads=4,
            cond_embed_dim=64,
            conditioning_type="cross_attention",
        )
        x = torch.randn(2, 20, 64)
        c = torch.randn(2, 64)
        out = block(x, c, num_frames=4, num_patches=5)
        assert out.shape == x.shape

    def test_gradient_flow_adaln(self):
        """Test gradients flow through AdaLN block."""
        block = DiTBlock(64, 4, conditioning_type="adaln", cond_embed_dim=64)
        x = torch.randn(2, 20, 64, requires_grad=True)
        c = torch.randn(2, 64, requires_grad=True)
        out = block(x, c, 4, 5)
        out.sum().backward()
        assert x.grad is not None
        assert c.grad is not None

    def test_gradient_flow_cross_attention(self):
        """Test gradients flow through cross-attention block."""
        block = DiTBlock(64, 4, conditioning_type="cross_attention", cond_embed_dim=64)
        x = torch.randn(2, 20, 64, requires_grad=True)
        c = torch.randn(2, 64, requires_grad=True)
        out = block(x, c, 4, 5)
        out.sum().backward()
        assert x.grad is not None
        assert c.grad is not None

    def test_different_mlp_ratios(self):
        """Test different MLP ratios."""
        for mlp_ratio in [2.0, 4.0]:
            block = DiTBlock(64, 4, mlp_ratio=mlp_ratio, cond_embed_dim=64)
            x = torch.randn(2, 20, 64)
            c = torch.randn(2, 64)
            out = block(x, c, 4, 5)
            assert out.shape == x.shape

    def test_with_dropout(self):
        """Test block has dropout layers configured."""
        block = DiTBlock(64, 4, dropout=0.5, attn_drop=0.5, cond_embed_dim=64)
        # Verify dropout layers exist in the MLP
        assert hasattr(block.mlp, "dropout")
        assert block.mlp.dropout.p == 0.5
        # Verify attention has dropout probability stored
        assert block.attn.spatial_attn.attn_drop_p == 0.5
        assert block.attn.temporal_attn.attn_drop_p == 0.5

    def test_eval_mode_deterministic(self):
        """Test block is deterministic in eval mode."""
        block = DiTBlock(64, 4, dropout=0.1, cond_embed_dim=64)
        block.eval()
        x = torch.randn(2, 20, 64)
        c = torch.randn(2, 64)
        out1 = block(x, c, 4, 5)
        out2 = block(x, c, 4, 5)
        assert torch.allclose(out1, out2)


class TestFinalLayer:
    """Tests for FinalLayer."""

    def test_output_shape(self):
        """Test final layer unpatchifies correctly."""
        final = FinalLayer(
            hidden_dim=64,
            out_channels=3,
            patch_size=(2, 2),
            cond_embed_dim=64,
        )
        # Input: tokens from 4 frames, 4x4 patches (16 patches per frame)
        x = torch.randn(2, 4 * 4 * 4, 64)
        c = torch.randn(2, 64)
        out = final(x, c, num_frames=4, num_patches_h=4, num_patches_w=4)
        # Output: (B, C, T, H, W) = (2, 3, 4, 8, 8)
        assert out.shape == (2, 3, 4, 8, 8)

    def test_different_patch_sizes(self):
        """Test different patch sizes."""
        for ph, pw in [(2, 2), (4, 4)]:
            final = FinalLayer(64, 3, (ph, pw), 64)
            # 4 frames, 4x4 patches
            x = torch.randn(2, 4 * 4 * 4, 64)
            c = torch.randn(2, 64)
            out = final(x, c, 4, 4, 4)
            assert out.shape == (2, 3, 4, 4 * ph, 4 * pw)

    def test_gradient_flow(self):
        """Test gradients flow through final layer."""
        final = FinalLayer(64, 3, (2, 2), 64)
        x = torch.randn(2, 80, 64, requires_grad=True)
        c = torch.randn(2, 64, requires_grad=True)
        out = final(x, c, 4, 4, 5)
        out.sum().backward()
        assert x.grad is not None
        assert c.grad is not None
