#!/bin/bash
#SBATCH --job-name=jit-full
#SBATCH --partition=pg_tata
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# JiT Full Training - 4x L40S Multi-GPU with DDP
# =============================================================================
#
# Trains ConditionalJiT + MarkerGNN + BioLoss + LPIPS for 250K steps.
# Uses torchrun for DDP with 4 L40S GPUs.
# Effective batch size: 4 per GPU x 4 GPUs = 16.
#
# Expected runtime: ~17-18 hours for 250K steps at ~4 steps/s on 4x L40S.
#
# Usage:
#   sbatch slurm/train_jit_multigpu.sbatch
#   sbatch slurm/train_jit_multigpu.sbatch max_steps=100000     # shorter run
#   sbatch slurm/train_jit_multigpu.sbatch lambda_lpips=0.05    # tune LPIPS
#   sbatch slurm/train_jit_multigpu.sbatch use_marker_gnn=false # ablation
#
# Resume from checkpoint:
#   sbatch slurm/train_jit_multigpu.sbatch resume_from=checkpoints/jit/step_0100000.pt
#
# Monitor:
#   squeue -u $USER
#   tail -f logs/jit-full_JOBID.out
#
# =============================================================================

set -e

echo "============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"
echo "============================================="

# Activate environment
source ~/.bashrc
conda activate multiplex 2>/dev/null || true

# Set paths
cd /orcd/home/002/tomli/multiplex

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
echo "  GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) x${SLURM_GPUS_ON_NODE}"
echo "============================================="

# Run multi-GPU training with torchrun
torchrun \
    --standalone \
    --nproc_per_node=${SLURM_GPUS_ON_NODE:-4} \
    scripts/train_jit.py \
    experiment_name="jit-full-250k" \
    max_steps=250000 \
    batch_size=4 \
    gradient_checkpointing=true \
    lambda_lpips=0.1 \
    lambda_bio=0.05 \
    use_marker_gnn=true \
    eval_every_n_steps=5000 \
    sample_every_n_steps=10000 \
    checkpoint_every_n_steps=25000 \
    log_every_n_steps=100 \
    "$@"

echo "============================================="
echo "Training complete!"
echo "End time: $(date)"
echo "Checkpoints: checkpoints/jit/"
echo "============================================="
