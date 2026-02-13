#!/bin/bash

### Task name
#SBATCH --account=your_account_here

### Job name
#SBATCH --job-name=name_of_your_job

### Output file (captures output before exec redirect takes over)
#SBATCH --output=results/00_slrm_logs/%x_%j.out

### Number of nodes
#SBATCH --nodes=1

### How many CPU cores to use
#SBATCH --ntasks-per-node=70

### How much memory in total (MB)
#SBATCH --mem=200G

### Mail notification configuration
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your_email_here

### Maximum runtime per task
#SBATCH --time=24:00:00

### set number of GPUs per task (v100, a100, h200)
#SBATCH --gres=gpu:a100:4
##SBATCH --constraint=a100_80gb


### Partition
#SBATCH --partition=gpu

### create time series, i.e. 100 jobs one after another. Each runs for 24 hours
##SBATCH --array=1-10%1


#####################################################################################
############################# Setup #################################################
#####################################################################################
# Create SBATCH output directory if it doesn't exist
mkdir -p results/00_slrm_logs
module load cuda/13.0.2


ENV_FILE="........./.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
else
    echo "Error: .env file not found at $ENV_FILE"
    echo "Please create a .env file with RESULTS_DIR and BASE_DIR defined."
    exit 1
fi

######################################################################################
############################# Set paths ##############################################
######################################################################################

sim_name="train" # name of the folder where you placed the yaml config

# Create log directory if it doesn't exist
log_dir="${RESULTS_DIR}/${sim_name}"
mkdir -p "$log_dir"

# Redirect all output to the run directory
log_file="${log_dir}/${sim_name}_${SLURM_JOB_ID}.out"
exec > "$log_file" 2>&1

python_exec="${BASE_DIR}/physicsflow/train/run_training.py"
config_file="${RESULTS_DIR}/${sim_name}/train.yaml"

nnodes=1
ngpus_per_node=2
export OMP_NUM_THREADS=1


#####################################################################################
############################# Training GPM ##########################################
#####################################################################################
echo "--------------------------------"
echo "Starting training..."
echo "config_file: $config_file"
echo "--------------------------------"

exec_args="--config_path $config_file"

# Capture Python output and errors in a variable and run the script

uv run torchrun --standalone --nproc_per_node=$ngpus_per_node $python_exec $exec_args
