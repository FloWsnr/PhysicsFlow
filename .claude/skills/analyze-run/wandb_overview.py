"""List all available metrics for a WandB run.

Connects to the WandB API and prints a structured overview of all logged
metrics, grouped by folder (train/, eval/, health/).

Usage:
    uv run python .claude/skills/analyze-run/wandb_overview.py --run_name <run_name>
    uv run python .claude/skills/analyze-run/wandb_overview.py --run_name <run_name> --env_file .env
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import wandb


def load_env(env_file: str) -> None:
    """Load environment variables from a .env file."""
    env_path = Path(env_file)
    if not env_path.exists():
        print(f"Warning: .env file not found at {env_path}")
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def load_wandb_config(run_name: str) -> dict:
    """Try to load wandb project/entity from the run's train.yaml."""
    results_dir = os.environ.get("RESULTS_DIR", "results")
    config_path = Path(results_dir) / run_name / "train.yaml"
    if not config_path.exists():
        return {}
    try:
        import yaml

        with open(config_path) as f:
            config = yaml.safe_load(f)
        return config.get("wandb", {})
    except Exception:
        return {}


def main():
    parser = argparse.ArgumentParser(description="List all available WandB metrics for a run")
    parser.add_argument("--run_name", required=True, help="WandB run ID (e.g. flow_001)")
    parser.add_argument("--project", default=None, help="WandB project name (default: from train.yaml)")
    parser.add_argument("--entity", default=None, help="WandB entity (default: from train.yaml)")
    parser.add_argument("--env_file", default=".env", help="Path to .env file (default: .env)")
    args = parser.parse_args()

    # Load environment
    load_env(args.env_file)

    # Get project/entity from config or args
    wandb_config = load_wandb_config(args.run_name)
    project = args.project or wandb_config.get("project", "physicsflow")
    entity = args.entity or wandb_config.get("entity")

    run_path = f"{entity}/{project}/{args.run_name}" if entity else f"{project}/{args.run_name}"

    print(f"Fetching metrics for: {run_path}")
    print("=" * 70)

    api = wandb.Api(timeout=30)
    try:
        run = api.run(run_path)
    except wandb.errors.CommError:
        print(f"Error: Run '{run_path}' not found.")
        sys.exit(1)

    # Print run metadata
    print(f"Run Name:    {run.name}")
    print(f"Run ID:      {run.id}")
    print(f"State:       {run.state}")
    print(f"Created:     {run.created_at}")
    if run.summary.get("_wandb", {}).get("runtime"):
        runtime_s = run.summary["_wandb"]["runtime"]
        hours, remainder = divmod(int(runtime_s), 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Runtime:     {hours}h {minutes}m {seconds}s")
    print(f"Tags:        {run.tags}")
    print(f"Notes:       {run.notes or '(none)'}")
    print()

    # Get all metric keys from history
    history_keys = set()
    try:
        # Scan a few rows to discover keys (metrics may appear at different steps)
        for i, row in enumerate(run.scan_history(page_size=10)):
            history_keys.update(row.keys())
            if i >= 9:
                break
    except Exception:
        pass

    # Also include summary keys
    summary_keys = set(run.summary.keys())

    all_keys = history_keys | summary_keys

    # Filter out internal wandb keys (prefixed with _)
    metric_keys = sorted(k for k in all_keys if not k.startswith("_"))

    # Group by folder
    grouped = defaultdict(list)
    for key in metric_keys:
        if "/" in key:
            folder, name = key.split("/", 1)
            grouped[folder].append(name)
        else:
            grouped["(ungrouped)"].append(key)

    # Print grouped metrics
    print("Available Metrics")
    print("-" * 70)

    # Print in a nice order: train, eval, health, then the rest
    priority_order = ["train", "eval", "health", "system"]
    sorted_folders = []
    for folder in priority_order:
        if folder in grouped:
            sorted_folders.append(folder)
    for folder in sorted(grouped.keys()):
        if folder not in priority_order:
            sorted_folders.append(folder)

    for folder in sorted_folders:
        metrics = grouped[folder]
        print(f"\n  {folder}/ ({len(metrics)} metrics)")
        for name in sorted(metrics):
            # Show latest value from summary if available
            full_key = f"{folder}/{name}" if folder != "(ungrouped)" else name
            val = run.summary.get(full_key)
            if val is not None and isinstance(val, (int, float)):
                print(f"    {name:<40s} (latest: {val:.6g})")
            else:
                print(f"    {name}")

    print()
    print(f"Total: {len(metric_keys)} metrics across {len(grouped)} groups")
    print()
    print("To fetch specific metrics, use:")
    print(f"  uv run python .claude/skills/analyze-run/wandb_fetch_metric.py \\")
    print(f"    --run_name {args.run_name} --metrics train/loss eval/loss")


if __name__ == "__main__":
    main()
