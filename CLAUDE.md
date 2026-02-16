# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PhysicsFlow is a generative modeling library for physics simulations using **Flow Matching**. It learns velocity fields to transport samples from noise to data distributions, with a focus on 5D physics data (C, T, H, W format).

## Development Commands

```bash
# Install dependencies (uses uv package manager)
uv sync --extra dev

# Run all tests
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_models/test_backbone/test_dit/test_dit_backbone.py

# Run training
uv run python physicsflow/train/run_training.py --config_path config/train.yaml
```

## Architecture

### Core Components

**Models** (`physicsflow/models/`):
- `FlowMatchingModel` - Main wrapper that combines a velocity backbone with a scheduler for flow matching
- `schedulers.py` - Defines interpolation paths between noise (x_0) and data (x_1)
- **Backbones** - Neural networks that predict velocity fields:
  - `DiTBackbone` - Diffusion Transformer with factorized spatio-temporal attention (preferred for production)
  - `MLPBackbone` - Simple MLP for testing

**Data** (`physicsflow/data/`):
- `PhysicsDataset` - Loads 5D physics data from HDF5 sources with normalization and trajectory sampling
- `WellDataset` - Base dataset from `the-well` library

**Training** (`physicsflow/train/`):
- `Trainer` - Distributed training with DDP, AMP (bfloat16), gradient checkpointing, W&B integration
- `Evaluator` - Model evaluation with ODE integration (Euler/midpoint methods)
- Configuration via YAML files in `config/`

### Data Flow
```
HDF5 → PhysicsDataset → DataLoader → FlowMatchingModel(DiTBackbone) → Trainer → Evaluator
```

## Key Patterns

- **Config-driven**: Hyperparameters in `config/train.yaml`, not hardcoded
- **Factory pattern**: `get_model()` creates models from config dicts
- **Dataclasses for outputs**: `FlowMatchingOutput`, `TrainingState`, `DiTConfig`
- **Continuous time**: Flow matching uses t ∈ [0,1]
- **Type hints required**: Full Python 3.13+ type annotations throughout
- **Einops for tensor ops**: Prefer `einops.rearrange` over manual reshape

## Dependencies

- PyTorch with CUDA 12.9 (configured via uv)
- `the-well` for physics datasets
- `wandb` for experiment tracking
- `einops` for tensor operations
