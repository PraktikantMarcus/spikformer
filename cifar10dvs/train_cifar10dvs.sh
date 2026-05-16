#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Spikformer CIFAR10-DVS training — SLURM job (single run, no chaining needed)
# Expected runtime: ~30 min on 8 H100s for 96 epochs
#
# Submission:
#   sbatch train_cifar10dvs.sh
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=spikformer_cifar10dvs
#SBATCH --partition=normal
#SBATCH --nodelist=multigpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1             # single task — torchrun spawns 8 processes internally
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=4:00:00
#SBATCH --output=/nfsscratch/fritzsche/cifar10dvs/logs/spikformer_cifar10dvs_%j.out
#SBATCH --error=/nfsscratch/fritzsche/cifar10dvs/logs/spikformer_cifar10dvs_%j.err

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR="/home/fritzsche/Spikformer/cifar10dvs"
DATA_DIR="/nfsscratch/fritzsche/cifar10dvs"
OUTPUT_DIR="/nfsscratch/fritzsche/spikformer_output/cifar10dvs"
LOG_DIR="/nfsscratch/fritzsche/cifar10dvs/logs"

# ── Environment setup ─────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

module purge
module load cuda/12.3
module load gnu12/12.3.0

source /home/fritzsche/qkformer/bin/activate

export LD_LIBRARY_PATH=$(python3 -c "import nvidia.cudnn, os; print(os.path.join(os.path.dirname(nvidia.cudnn.__file__), 'lib'))" 2>/dev/null):$LD_LIBRARY_PATH

echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Start:     $(date)"

# ── Resume detection ──────────────────────────────────────────────────────────
RESUME_ARG=""
# Checkpoints are saved as checkpoint_{epoch}.pth inside a nested output subdir.
# Pass --resume manually if you need to resume from a specific checkpoint.

# ── Training ──────────────────────────────────────────────────────────────────
cd "$PROJECT_DIR"

torchrun \
    --standalone \
    --nproc_per_node=8 \
    train.py \
    --model spikformer \
    --dataset cifar10dvs \
    --num-classes 10 \
    --data-path "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --device cuda \
    --T 16 \
    --epochs 96 \
    --batch-size 16 \
    --workers 8 \
    --opt adamw \
    --lr 1e-3 \
    --min-lr 1e-5 \
    --weight-decay 0.06 \
    --sched cosine \
    --warmup-epochs 10 \
    --cooldown-epochs 10 \
    --smoothing 0.1 \
    --mixup 0.5 \
    --mixup-prob 0.5 \
    --mixup-switch-prob 0.5 \
    --mixup-mode batch \
    --amp \
    $RESUME_ARG

echo "End: $(date)"
