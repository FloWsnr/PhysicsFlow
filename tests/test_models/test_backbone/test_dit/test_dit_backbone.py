"""Tests for DiT backbone."""

import pytest
import torch

from physicsflow.models.dit import (
    DiTBackbone,
    DiTConfig,
    get_dit_config,
    list_dit_configs,
)
from physicsflow.models.flow_matching import FlowMatchingModel
from physicsflow.models.flow_matching.schedulers import (
    CondOTScheduler,
    CosineScheduler,
    LinearVPScheduler,
    PolynomialScheduler,
)


class TestDiTConfig:
    """Tests for DiT configurations."""

    @pytest.mark.parametrize("size", ["DiT-S", "DiT-B", "DiT-L", "DiT-XL"])
    def test_config_creation(self, size):
        """Test all config sizes can be retrieved."""
        config = get_dit_config(size)
        assert config.hidden_dim > 0
        assert config.depth > 0
        assert config.num_heads > 0
        assert config.hidden_dim % config.num_heads == 0

    def test_invalid_config(self):
        """Test error on invalid config name."""
        with pytest.raises(ValueError):
            get_dit_config("DiT-Invalid")

    def test_list_configs(self):
        """Test listing available configs."""
        configs = list_dit_configs()
        assert "DiT-S" in configs
        assert "DiT-B" in configs
        assert "DiT-L" in configs
        assert "DiT-XL" in configs

    def test_config_head_dim(self):
        """Test head dimension property."""
        config = DiTConfig(hidden_dim=256, depth=6, num_heads=8)
        assert config.head_dim == 32

    def test_config_validation(self):
        """Test config validation for invalid hidden_dim/num_heads."""
        with pytest.raises(ValueError):
            DiTConfig(hidden_dim=256, depth=6, num_heads=7)  # Not divisible


class TestDiTBackbone:
    """Tests for DiTBackbone."""

    @pytest.fixture
    def dit_model(self):
        """Create a small DiT for testing."""
        return DiTBackbone(
            in_channels=3,
            spatial_size=(32, 32),
            temporal_size=8,
            cond_dim=5,
            hidden_dim=128,
            depth=2,
            num_heads=4,
            patch_size=(4, 4),
        )

    @pytest.fixture
    def dit_model_no_cond(self):
        """Create a small DiT without conditioning."""
        return DiTBackbone(
            in_channels=3,
            spatial_size=(32, 32),
            temporal_size=8,
            cond_dim=0,
            hidden_dim=128,
            depth=2,
            num_heads=4,
            patch_size=(4, 4),
        )

    def test_output_shape(self, dit_model):
        """Test output matches input shape."""
        x_t = torch.randn(2, 3, 8, 32, 32)
        t = torch.rand(2)
        cond = torch.randn(2, 5)
        out = dit_model(x_t, t, cond)
        assert out.shape == x_t.shape

    def test_without_conditioning(self, dit_model_no_cond):
        """Test model works without physics parameters."""
        x_t = torch.randn(2, 3, 8, 32, 32)
        t = torch.rand(2)
        out = dit_model_no_cond(x_t, t, None)
        assert out.shape == x_t.shape

    def test_gradient_flow(self, dit_model):
        """Test gradients flow through backbone."""
        x_t = torch.randn(2, 3, 8, 32, 32, requires_grad=True)
        t = torch.rand(2)
        cond = torch.randn(2, 5)
        out = dit_model(x_t, t, cond)
        out.sum().backward()
        assert x_t.grad is not None
        assert x_t.grad.abs().sum() > 0

    def test_gradient_checkpointing(self):
        """Test gradient checkpointing works."""
        model = DiTBackbone(
            in_channels=3,
            spatial_size=(32, 32),
            temporal_size=8,
            cond_dim=5,
            hidden_dim=128,
            depth=2,
            num_heads=4,
            patch_size=(4, 4),
            use_gradient_checkpointing=True,
        )
        model.train()
        x_t = torch.randn(2, 3, 8, 32, 32, requires_grad=True)
        t = torch.rand(2)
        cond = torch.randn(2, 5)
        out = model(x_t, t, cond)
        out.sum().backward()
        assert x_t.grad is not None

    def test_conditioning_types(self):
        """Test both conditioning mechanisms."""
        for cond_type in ["adaln", "cross_attention"]:
            model = DiTBackbone(
                in_channels=3,
                spatial_size=(32, 32),
                temporal_size=8,
                cond_dim=5,
                hidden_dim=128,
                depth=2,
                num_heads=4,
                patch_size=(4, 4),
                conditioning_type=cond_type,
            )
            x_t = torch.randn(2, 3, 8, 32, 32)
            t = torch.rand(2)
            cond = torch.randn(2, 5)
            out = model(x_t, t, cond)
            assert out.shape == x_t.shape

    def test_from_config_string(self):
        """Test creating from config string."""
        model = DiTBackbone.from_config(
            "DiT-S",
            in_channels=3,
            spatial_size=(64, 64),
            temporal_size=10,
            cond_dim=5,
        )
        x_t = torch.randn(1, 3, 10, 64, 64)
        t = torch.rand(1)
        cond = torch.randn(1, 5)
        out = model(x_t, t, cond)
        assert out.shape == x_t.shape

    def test_from_config_object(self):
        """Test creating from config object."""
        config = DiTConfig(hidden_dim=256, depth=4, num_heads=8)
        model = DiTBackbone.from_config(
            config,
            in_channels=3,
            spatial_size=(32, 32),
            temporal_size=8,
            cond_dim=5,
        )
        assert model.hidden_dim == 256
        assert len(model.blocks) == 4

    def test_from_config_with_overrides(self):
        """Test config overrides."""
        model = DiTBackbone.from_config(
            "DiT-S",
            in_channels=3,
            spatial_size=(32, 32),
            temporal_size=8,
            cond_dim=5,
            hidden_dim=512,  # Override
            depth=6,  # Override
        )
        assert model.hidden_dim == 512
        assert len(model.blocks) == 6

    def test_get_num_params(self, dit_model):
        """Test parameter counting."""
        total_params = dit_model.get_num_params()
        non_embed_params = dit_model.get_num_params(non_embedding=True)
        assert total_params > 0
        assert non_embed_params < total_params

    def test_extra_repr(self, dit_model):
        """Test string representation."""
        repr_str = dit_model.extra_repr()
        assert "in_channels=3" in repr_str
        assert "hidden_dim=128" in repr_str
        assert "params=" in repr_str

    def test_invalid_spatial_size(self):
        """Test error on spatial size not divisible by patch size."""
        with pytest.raises(ValueError):
            DiTBackbone(
                in_channels=3,
                spatial_size=(30, 30),  # Not divisible by 4
                temporal_size=8,
                cond_dim=5,
                hidden_dim=128,
                depth=2,
                num_heads=4,
                patch_size=(4, 4),
            )

    def test_different_input_channels(self):
        """Test different input channel counts."""
        for channels in [1, 3, 5, 16]:
            model = DiTBackbone(
                in_channels=channels,
                spatial_size=(32, 32),
                temporal_size=8,
                cond_dim=5,
                hidden_dim=128,
                depth=2,
                num_heads=4,
                patch_size=(4, 4),
            )
            x_t = torch.randn(2, channels, 8, 32, 32)
            t = torch.rand(2)
            cond = torch.randn(2, 5)
            out = model(x_t, t, cond)
            assert out.shape == x_t.shape

    def test_eval_mode_deterministic(self, dit_model):
        """Test model is deterministic in eval mode."""
        dit_model.eval()
        x_t = torch.randn(2, 3, 8, 32, 32)
        t = torch.rand(2)
        cond = torch.randn(2, 5)
        out1 = dit_model(x_t, t, cond)
        out2 = dit_model(x_t, t, cond)
        assert torch.allclose(out1, out2)


class TestDiTWithFlowMatching:
    """Integration tests with FlowMatchingModel."""

    @pytest.fixture
    def flow_model(self):
        """Create FlowMatchingModel with DiT backbone."""
        dit = DiTBackbone(
            in_channels=3,
            spatial_size=(32, 32),
            temporal_size=8,
            cond_dim=5,
            hidden_dim=128,
            depth=2,
            num_heads=4,
            patch_size=(4, 4),
        )
        return FlowMatchingModel(dit, scheduler=CondOTScheduler())

    @pytest.fixture
    def sample_data(self):
        """Sample data for testing."""
        return {
            "input_fields": torch.randn(2, 3, 8, 32, 32),
            "constant_scalars": torch.randn(2, 5),
        }

    def test_forward_pass(self, flow_model, sample_data):
        """Test DiT works with FlowMatchingModel forward."""
        x_1 = sample_data["input_fields"]
        cond = sample_data["constant_scalars"]
        output = flow_model(x_1, cond)

        assert output.predicted_velocity.shape == x_1.shape
        assert output.target_velocity.shape == x_1.shape
        assert output.x_t.shape == x_1.shape

    def test_loss_computation(self, flow_model, sample_data):
        """Test loss can be computed from output."""
        x_1 = sample_data["input_fields"]
        cond = sample_data["constant_scalars"]
        output = flow_model(x_1, cond)

        loss = torch.nn.functional.mse_loss(
            output.predicted_velocity, output.target_velocity
        )
        assert loss.shape == ()
        assert loss.requires_grad

    def test_training_step(self, flow_model, sample_data):
        """Test full training step with DiT."""
        optimizer = torch.optim.Adam(flow_model.parameters(), lr=1e-4)

        x_1 = sample_data["input_fields"]
        cond = sample_data["constant_scalars"]

        output = flow_model(x_1, cond)
        loss = torch.nn.functional.mse_loss(
            output.predicted_velocity, output.target_velocity
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Should complete without error
        assert loss.item() > 0

    def test_sampling(self, flow_model):
        """Test sample generation with DiT."""
        shape = (2, 3, 8, 32, 32)
        cond = torch.randn(2, 5)
        samples = flow_model.sample(shape, cond, num_steps=5)
        assert samples.shape == shape

    def test_sampling_methods(self, flow_model):
        """Test different sampling methods."""
        shape = (2, 3, 8, 32, 32)
        cond = torch.randn(2, 5)

        torch.manual_seed(42)
        euler = flow_model.sample(shape, cond, num_steps=5, method="euler")

        torch.manual_seed(42)
        midpoint = flow_model.sample(shape, cond, num_steps=5, method="midpoint")

        assert euler.shape == shape
        assert midpoint.shape == shape
        # Different methods should give different results
        assert not torch.allclose(euler, midpoint)

    def test_different_schedulers(self, sample_data):
        """Test DiT with different flow matching schedulers."""
        dit = DiTBackbone(
            in_channels=3,
            spatial_size=(32, 32),
            temporal_size=8,
            cond_dim=5,
            hidden_dim=128,
            depth=2,
            num_heads=4,
            patch_size=(4, 4),
        )

        x_1 = sample_data["input_fields"]
        cond = sample_data["constant_scalars"]

        for scheduler in [CondOTScheduler(), CosineScheduler(), LinearVPScheduler(), PolynomialScheduler()]:
            model = FlowMatchingModel(dit, scheduler=scheduler)
            output = model(x_1, cond)
            loss = torch.nn.functional.mse_loss(
                output.predicted_velocity, output.target_velocity
            )
            assert loss.shape == ()
            assert torch.isfinite(loss)


class TestDiTModelFactory:
    """Tests for model factory with DiT."""

    @staticmethod
    def _base_config(**overrides):
        """Base config with all required params."""
        config = {
            "in_channels": 3,
            "spatial_size": [32, 32],
            "temporal_size": 8,
            "cond_dim": 5,
            "hidden_dim": 384,
            "depth": 12,
            "num_heads": 6,
            "mlp_ratio": 4.0,
            "patch_size": [4, 4],
            "time_embed_dim": 256,
            "conditioning_type": "adaln",
            "dropout": 0.0,
            "attn_drop": 0.0,
            "learnable_pos_embed": True,
            "gradient_checkpointing": False,
            "scheduler": {"type": "cond_ot", "params": {}},
        }
        config.update(overrides)
        return config

    def test_factory_creates_dit(self):
        """Test factory creates DiT backbone."""
        from physicsflow.models.model_utils import get_model

        model = get_model(self._base_config())

        assert isinstance(model, FlowMatchingModel)
        assert isinstance(model.velocity_net, DiTBackbone)

    def test_factory_with_overrides(self):
        """Test factory with config overrides."""
        from physicsflow.models.model_utils import get_model

        model = get_model(self._base_config(hidden_dim=512, depth=6))

        assert model.velocity_net.hidden_dim == 512
        assert len(model.velocity_net.blocks) == 6

    def test_factory_with_cross_attention(self):
        """Test factory creates cross-attention DiT."""
        from physicsflow.models.model_utils import get_model

        model = get_model(self._base_config(conditioning_type="cross_attention"))

        x_1 = torch.randn(2, 3, 8, 32, 32)
        cond = torch.randn(2, 5)
        output = model(x_1, cond)
        assert output.predicted_velocity.shape == x_1.shape
