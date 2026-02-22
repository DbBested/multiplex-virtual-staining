#!/bin/bash
#SBATCH --job-name=jit-flow
#SBATCH --partition=pg_tata
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# JiT Flow Matching Training - ConditionalJiT with velocity prediction
# =============================================================================
#
# Trains ConditionalJiT with flow matching on DeepLIIF dataset.
# Single GPU (L40S), step-based training with EMA.
#
# Usage:
#   sbatch slurm/train_jit.sbatch                    # 10K stability run
#   sbatch slurm/train_jit.sbatch max_steps=50000    # full training
#
# Monitor:
#   squeue -u $USER
#   tail -f logs/jit-flow_JOBID.out
#
# =============================================================================

set -e

echo "============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Start time: $(date)"
echo "============================================="

# Activate environment
source ~/.bashrc
conda activate multiplex 2>/dev/null || true

# Set paths
cd /orcd/home/002/tomli/multiplex
export WANDB_MODE=disabled  # Enable if you have W&B setup

# Create output directories
mkdir -p logs
mkdir -p checkpoints/jit

# Print environment
echo "============================================="
echo "Environment:"
echo "  Data: data/deepliif"
echo "  Checkpoints: checkpoints/jit"
echo "  Python: $(which python)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "============================================="

# Run training
# Default: 10K step stability validation
# Override via CLI: sbatch slurm/train_jit.sbatch max_steps=50000
python scripts/train_jit.py \
    experiment_name="jit-flow-10k-stability" \
    max_steps=10000 \
    eval_every_n_steps=1000 \
    sample_every_n_steps=2000 \
    checkpoint_every_n_steps=5000 \
    log_every_n_steps=50 \
    "$@"

echo "============================================="
echo "Training complete!"
echo "End time: $(date)"
echo "Checkpoints: checkpoints/jit/"
echo "============================================="
