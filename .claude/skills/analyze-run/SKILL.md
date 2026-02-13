---
name: analyze-run
description: Analyze a previous training run by scanning logs, viewing eval images, and fetching WandB metrics
---

# Analyze Training Run

Use this skill to analyze a completed or in-progress training run. The analysis covers three areas:
1. SLURM log inspection
2. Eval image review (from epoch folders)
3. WandB metric curves

## Step 1: Identify the Run

Ask the user which run to analyze if not specified. Runs are stored in the results directory:
```bash
ls results/
```

Each run folder contains:
- `train.yaml` - training configuration
- `epoch_XXXX/` - epoch checkpoints and eval images
- `wandb/` - local wandb logs
- `latest.pt`, `best.pt` - checkpoint files

SLURM logs are in `results/00_slrm_logs/` named as `<run_name>_<job_id>.out`.

## Step 2: Scan SLURM Logs

Find and read the SLURM log for the run:
```bash
ls results/00_slrm_logs/<run_name>_*.out
```

Look for:
- **Training progress**: Epoch/update lines showing loss metrics
- **Loss spikes**: Lines containing "Loss spike detected"
- **Errors/warnings**: Any error messages, OOM errors, CUDA errors
- **Evaluation results**: Lines from "Evaluator" showing validation metrics
- **Checkpoints**: "Saved latest/best checkpoint" messages
- **Health issues**: NaN detections
- **Timing**: First and last timestamps to calculate total runtime
- **WandB URL**: Usually near the top, e.g. `wandb: View run at https://wandb.ai/...`

Read the **beginning** (setup info, config), **end** (final metrics, completion status), and scan for warnings/errors throughout.

Summarize:
- Current training status (running, completed, crashed)
- Total updates completed / total planned
- Latest training metrics
- Number of loss spikes or anomalies
- Any errors or warnings

## Step 3: Review Eval Images

Check which epoch folders exist and view the most recent validation sample images:
```bash
ls results/<run_name>/epoch_*/
```

Each epoch folder contains validation sample images showing reconstruction quality. Read the **most recent 2-3 epoch** image files to visually assess reconstruction quality and improvement over time.

Compare images across epochs to comment on:
- Overall reconstruction quality
- Whether fine details are being captured
- Any artifacts or blurriness
- Visual improvement trend

## Step 4: Fetch WandB Metrics

### 4a: Get Metric Overview

Run the overview script to see all available metrics:
```bash
uv run python .claude/skills/analyze-run/wandb_overview.py --run_name <run_name>
```

This shows all logged metrics grouped by folder (train/, eval/, system/) with their latest values.

### 4b: Fetch Key Metrics

Fetch the most important training curves. Common metrics to check:

**Training metrics:**
```bash
uv run python .claude/skills/analyze-run/wandb_fetch_metric.py \
    --run_name <run_name> --metrics "train/*" --samples 100
```

**Eval metrics:**
```bash
uv run python .claude/skills/analyze-run/wandb_fetch_metric.py \
    --run_name <run_name> --metrics "eval/*" --samples 100
```

You can also fetch specific metrics:
```bash
uv run python .claude/skills/analyze-run/wandb_fetch_metric.py \
    --run_name <run_name> --metrics train/loss eval/loss --samples 50
```

Or save to CSV for further analysis:
```bash
uv run python .claude/skills/analyze-run/wandb_fetch_metric.py \
    --run_name <run_name> --metrics "train/*" --format csv --output metrics.csv
```

## Step 5: Synthesize Findings

After gathering all information, provide a structured summary:

1. **Run Overview**: Model type, size, dataset, total updates, runtime
2. **Training Progress**: Loss curves trend, convergence assessment
3. **Reconstruction Quality**: Visual assessment from eval images
4. **Health Assessment**: Loss spikes, gradient norms, any anomalies
5. **Recommendations**: Whether to continue training, adjust hyperparameters, or flag issues
