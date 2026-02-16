# PhysicsFlow

A generative modeling library for physics simulations using Flow Matching.

## Overview

PhysicsFlow learns velocity fields to transport samples from noise to data distributions. It is designed for 5D physics data (channels, time, height, width) and supports distributed training with automatic mixed precision.

## Features

- Flow Matching with configurable interpolation schedulers
- DiT (Diffusion Transformer) backbone with factorized spatio-temporal attention
- HDF5 data loading with normalization and trajectory sampling
- Distributed training (DDP) with gradient checkpointing
- Weights & Biases integration for experiment tracking

## Installation

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

For development (includes pytest and other dev tools):

```bash
uv sync --extra dev
```

## Usage

```bash
# Training
uv run python physicsflow/train/run_training.py --config_path config/train.yaml

# Run tests
uv run pytest tests/
```

## Configuration

Training parameters are specified in YAML config files. See `config/train.yaml` for available options including dataset settings, model architecture, optimizer, and learning rate scheduling.

## License

See LICENSE file.
