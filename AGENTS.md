# Repository Guidelines

## Project Structure & Module Organization
`physicsflow/` contains the library code:
- `physicsflow/data/`: dataset and dataloader logic (HDF5 + preprocessing)
- `physicsflow/models/`: DiT backbone, flow-matching model, and shared model utilities
- `physicsflow/train/`: training entrypoint, evaluation, optimizer/scheduler/checkpoint utilities

`tests/` mirrors runtime modules (`test_data/`, `test_models/`, `test_train/`) and uses shared fixtures in `tests/conftest.py`.  
`config/train.yaml` is the main training configuration.  
`scripts/train_riv.sh` is a SLURM launch template.  
`results/` stores run outputs (logs/checkpoints), not source code.

## Build, Test, and Development Commands
- `uv sync --extra dev`: install runtime + developer dependencies.
- `uv run pytest tests/`: run full test suite.
- `uv run pytest tests/test_train/test_eval.py`: run a targeted test module during iteration.
- `uv run python physicsflow/train/run_training.py --config_path config/train.yaml`: single-process training.
- `uv run torchrun --standalone --nproc_per_node=2 physicsflow/train/run_training.py --config_path <train.yaml>`: multi-GPU local run.

## Coding Style & Naming Conventions
- Python 3.13+ with 4-space indentation and explicit type hints.
- Use `snake_case` for modules/functions/variables, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants.
- Keep behavior config-driven via YAML (avoid hardcoded experiment parameters).
- Prefer small, focused functions with clear validation and actionable error messages.
- No enforced formatter/linter is configured in `pyproject.toml`; keep style PEP 8/Black-compatible.

## Testing Guidelines
- Framework: `pytest`.
- Test files follow `test_<feature>.py` and should mirror package structure where practical.
- Add or update tests for every behavioral change; bug fixes should include regression tests.
- Run targeted tests first, then `uv run pytest tests/` before opening a PR.
- No explicit coverage threshold is configured; maintain or improve coverage in touched areas.

## Commit & Pull Request Guidelines
- Prefer imperative commit subjects, with optional scoped prefixes seen in history (for example, `fix(eval): validate eval_fraction`).
- Keep commits focused and descriptive; avoid vague messages.
- PRs should include:
  - what changed and why
  - linked issue/experiment context
  - test evidence (commands run)
  - config/runtime impact notes (env vars, GPU/distributed assumptions, checkpoint compatibility)

## Configuration & Secrets
- `.env` is used for local paths/secrets (for example `DATA_DIR`; SLURM scripts also reference `RESULTS_DIR` and `BASE_DIR`).
- Never commit secrets, dataset dumps, or large checkpoints.
