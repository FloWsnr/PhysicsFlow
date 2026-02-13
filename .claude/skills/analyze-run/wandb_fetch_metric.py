"""Fetch specific metrics from a WandB run and display as a table or CSV.

Downloads the time-series data for one or more metrics and prints them.
Can output as a formatted table, CSV, or JSON. Does not require pandas.

Usage:
    # Fetch specific metrics as table
    uv run python .claude/skills/analyze-run/wandb_fetch_metric.py \
        --run_name <run_name> --metrics train/loss eval/loss

    # Output as CSV
    uv run python .claude/skills/analyze-run/wandb_fetch_metric.py \
        --run_name <run_name> --metrics train/loss --format csv

    # Limit number of samples (evenly spaced)
    uv run python .claude/skills/analyze-run/wandb_fetch_metric.py \
        --run_name <run_name> --metrics train/loss eval/loss --samples 50

    # Fetch all metrics matching a pattern
    uv run python .claude/skills/analyze-run/wandb_fetch_metric.py \
        --run_name <run_name> --metrics "train/*" --samples 20

    # Save to file
    uv run python .claude/skills/analyze-run/wandb_fetch_metric.py \
        --run_name <run_name> --metrics train/loss --format csv --output metrics.csv
"""

import argparse
import csv
import fnmatch
import json
import os
import sys
from pathlib import Path

import wandb


def load_env(env_file: str) -> None:
    """Load environment variables from a .env file."""
    env_path = Path(env_file)
    if not env_path.exists():
        print(f"Warning: .env file not found at {env_path}", file=sys.stderr)
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


def discover_all_keys(run) -> set[str]:
    """Discover all metric keys from a run."""
    all_keys = set()
    try:
        for row in run.scan_history(page_size=1):
            all_keys.update(row.keys())
            break
    except Exception:
        pass
    all_keys.update(run.summary.keys())
    return {k for k in all_keys if not k.startswith("_")}


def resolve_glob_metrics(run, metric_patterns: list[str]) -> list[str]:
    """Expand glob patterns (e.g. 'train/*') into actual metric keys."""
    all_keys = discover_all_keys(run)

    resolved = []
    for pattern in metric_patterns:
        if "*" in pattern or "?" in pattern:
            matches = sorted(k for k in all_keys if fnmatch.fnmatch(k, pattern))
            if not matches:
                print(f"Warning: no metrics matched pattern '{pattern}'", file=sys.stderr)
            resolved.extend(matches)
        else:
            resolved.append(pattern)
    return resolved


def fetch_history(run, metrics: list[str], max_samples: int) -> list[dict]:
    """Fetch metric history using scan_history (no pandas dependency).

    Since different metrics may be logged at different steps (e.g. train/ vs eval/),
    we fetch each metric group separately and merge by step. Sparse metrics (like
    eval, logged once per epoch) are always preserved during downsampling.
    """
    # Group metrics by their folder prefix to batch related metrics
    groups: dict[str, list[str]] = {}
    for m in metrics:
        prefix = m.split("/")[0] if "/" in m else ""
        groups.setdefault(prefix, []).append(m)

    # Fetch each group separately, track row counts per group
    step_data: dict[int, dict] = {}  # step -> {metric: value, ...}
    group_counts: dict[str, int] = {}

    for prefix, group_metrics in groups.items():
        keys_to_fetch = list(set(group_metrics + ["_step"]))
        count = 0
        for row in run.scan_history(keys=keys_to_fetch):
            step = row.get("_step", 0)
            if step not in step_data:
                step_data[step] = {"_step": step}
            for m in group_metrics:
                if m in row and row[m] is not None:
                    step_data[step][m] = row[m]
            count += 1
        group_counts[prefix] = count

    if not step_data:
        return []

    # Convert to sorted list
    rows = sorted(step_data.values(), key=lambda r: r.get("_step", 0))

    # Identify sparse metrics (groups with significantly fewer rows than the largest)
    sparse_metrics = set()
    if group_counts:
        max_count = max(group_counts.values())
        for prefix, count in group_counts.items():
            if count < max_count * 0.1:  # Less than 10% of the densest group
                sparse_metrics.update(groups[prefix])

    # Downsample if needed, preserving rows with sparse metrics
    if len(rows) > max_samples and sparse_metrics:
        # Split into must-keep (has sparse metrics) and can-downsample
        must_keep = []
        can_downsample = []
        for row in rows:
            if any(m in row for m in sparse_metrics):
                must_keep.append(row)
            else:
                can_downsample.append(row)

        # Downsample the rest to fill remaining budget
        remaining = max(max_samples - len(must_keep), 1)
        if len(can_downsample) > remaining:
            step_size = len(can_downsample) / remaining
            indices = [int(i * step_size) for i in range(remaining)]
            if indices[-1] != len(can_downsample) - 1:
                indices[-1] = len(can_downsample) - 1
            can_downsample = [can_downsample[i] for i in indices]

        rows = sorted(must_keep + can_downsample, key=lambda r: r.get("_step", 0))

    elif len(rows) > max_samples:
        step_size = len(rows) / max_samples
        indices = [int(i * step_size) for i in range(max_samples)]
        if indices[-1] != len(rows) - 1:
            indices[-1] = len(rows) - 1
        rows = [rows[i] for i in indices]

    return rows


def main():
    parser = argparse.ArgumentParser(description="Fetch specific metrics from a WandB run")
    parser.add_argument("--run_name", required=True, help="WandB run ID (e.g. flow_001)")
    parser.add_argument("--metrics", required=True, nargs="+", help="Metric keys to fetch (e.g. train/loss eval/loss)")
    parser.add_argument("--project", default=None, help="WandB project name (default: from train.yaml)")
    parser.add_argument("--entity", default=None, help="WandB entity (default: from train.yaml)")
    parser.add_argument("--env_file", default=".env", help="Path to .env file (default: .env)")
    parser.add_argument("--samples", type=int, default=500, help="Max number of data points to fetch (default: 500)")
    parser.add_argument("--format", choices=["table", "csv", "json"], default="table", help="Output format (default: table)")
    parser.add_argument("--output", default=None, help="Output file path (default: stdout)")
    args = parser.parse_args()

    # Load environment
    load_env(args.env_file)

    # Get project/entity from config or args
    wandb_config = load_wandb_config(args.run_name)
    project = args.project or wandb_config.get("project", "physicsflow")
    entity = args.entity or wandb_config.get("entity")

    run_path = f"{entity}/{project}/{args.run_name}" if entity else f"{project}/{args.run_name}"

    api = wandb.Api(timeout=30)
    try:
        run = api.run(run_path)
    except wandb.errors.CommError:
        print(f"Error: Run '{run_path}' not found.", file=sys.stderr)
        sys.exit(1)

    # Resolve glob patterns
    metrics = resolve_glob_metrics(run, args.metrics)
    if not metrics:
        print("Error: No metrics to fetch.", file=sys.stderr)
        sys.exit(1)

    print(f"Fetching {len(metrics)} metrics from {run_path}...", file=sys.stderr)

    # Fetch history
    rows = fetch_history(run, metrics, args.samples)

    if not rows:
        print("No data found for the requested metrics.", file=sys.stderr)
        sys.exit(1)

    # Determine columns present in data
    columns = ["_step"] + [m for m in metrics if any(m in row for row in rows)]
    missing = set(metrics) - {m for m in metrics if any(m in row for row in rows)}
    if missing:
        print(f"Warning: metrics not found in history: {', '.join(sorted(missing))}", file=sys.stderr)

    # Output
    out_file = open(args.output, "w") if args.output else sys.stdout

    try:
        if args.format == "csv":
            writer = csv.DictWriter(out_file, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({c: row.get(c, "") for c in columns})
            if args.output:
                print(f"Saved to {args.output}", file=sys.stderr)

        elif args.format == "json":
            clean_rows = [{c: row.get(c) for c in columns if row.get(c) is not None} for row in rows]
            json_str = json.dumps(clean_rows, indent=2, default=str)
            print(json_str, file=out_file)
            if args.output:
                print(f"Saved to {args.output}", file=sys.stderr)

        else:
            # Table format
            print(f"\nMetrics for run: {args.run_name}", file=out_file)
            print(f"Data points: {len(rows)}", file=out_file)
            print("=" * (14 * len(columns) + 2), file=out_file)

            # Header
            header = ""
            for col in columns:
                label = col.replace("_step", "step")
                header += f"{label:>14s}"
            print(header, file=out_file)
            print("-" * (14 * len(columns) + 2), file=out_file)

            # Rows
            for row in rows:
                line = ""
                for col in columns:
                    val = row.get(col)
                    if val is None:
                        line += f"{'':>14s}"
                    elif col == "_step":
                        line += f"{int(val):>14d}"
                    elif isinstance(val, float):
                        line += f"{val:>14.6f}"
                    else:
                        line += f"{str(val):>14s}"
                print(line, file=out_file)

            if args.output:
                print(f"Saved to {args.output}", file=sys.stderr)
    finally:
        if args.output and out_file is not sys.stdout:
            out_file.close()


if __name__ == "__main__":
    main()
