---
name: slurm
description: Submit and manage SLURM jobs on the Rivanna HPC cluster for training runs
---

# SLURM Job Submission

## Setting Up a New Training Run

Do the following steps, if no script or config is present yet.
Sometimes, the files are already present. In that case, check if the slurm commands and config match up.

1. Create a results folder for your experiment:
   ```bash
   mkdir -p results/<experiment_name>
   ```

2. Copy the template SLURM script:
   ```bash
   cp physicsflow/train/scripts/train_riv.sh results/<experiment_name>/
   ```
3. Edit the SLURM script to customize:
   - `--job-name`: Set to your experiment name
   - `--output`: Update log path to match experiment name
   - `sim_name`: Set to your experiment folder name
   - `ENV_FILE="........./.env"`: Fill in the real .env (present in the top dir) path here

4. Copy the template config (`config/train.yaml`). Adapt:
   - `time_limit`: we usually do 72h (max), but might do less for test runs
   - model and backbone settings
   - dataloader workers: we usually do 16 workers per GPU, but might increase or decrease.
   - `total_updates`: usually at least 200k steps.
   - Check other hyperparameters as needed.

5. Adapt the number of GPUs (bash script) and batch size (config) depending on the model size.
   - choose an appropriate number and variant of GPU. For this, check which GPUs and RAM are available with ``qlist -p gpu``. Make sure the model fits on the GPU.
   - depending on the number of GPUs (and corresponding dataloaders), adapt `ntasks-per-node`. Make sure to add one core per GPU as main process.

## SBATCH Directives (Rivanna)

```bash
#SBATCH --account=sds_baek_energetic    # Allocation account
#SBATCH --job-name=<experiment_name>    # Job name (shows in squeue)
#SBATCH --output=results/00_slrm_logs/<experiment_name>_%j.out  # Log file (%j = job ID)
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks-per-node=68            # CPU cores
#SBATCH --mem=400G                      # Total memory
#SBATCH --time=72:00:00                 # Max runtime (HH:MM:SS)
#SBATCH --gres=gpu:a100:4               # GPU type and count
#SBATCH --constraint=a100_80gb          # Use 80GB A100s (optional)
#SBATCH --partition=gpu                 # GPU partition
#SBATCH --mail-type=ALL                 # Email notifications
#SBATCH --mail-user=<your_email>        # Your email
```

## Available GPU Options

- `gpu:a40:N` - NVIDIA A40 (40GB)
- `gpu:a100:N` - NVIDIA A100 (40GB or 80GB)
- `gpu:h200:N` - NVIDIA H200

Add `--constraint=a100_80gb` for 80GB A100s specifically.

## Common Commands

```bash
# Submit job
sbatch <script.sh>

# Check your jobs
squeue -u $USER

# Check all GPU jobs
squeue -p gpu

# Get detailed job info
scontrol show job <job_id>

# Cancel a job
scancel <job_id>

# Cancel all your jobs
scancel -u $USER

# Check job history/accounting
sacct -j <job_id>

# Check cluster status
qlist -p gpu
```

## Monitoring Output

```bash
# Watch job output in real-time
tail -f results/00_slrm_logs/<job_name>_<job_id>.out

# Check recent output
tail -100 results/00_slrm_logs/<job_name>_<job_id>.out
```

## Troubleshooting

- **Job pending**: Check `squeue -u $USER` for reason (Resources, Priority, etc.)
- **Job failed**: Check the `.out` log file in `results/00_slrm_logs/`
- **OOM errors**: Reduce batch size or request more memory
- **CUDA errors**: Check GPU availability with `qlist -p gpu`
