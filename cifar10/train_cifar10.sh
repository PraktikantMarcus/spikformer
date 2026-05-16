#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Spikformer CIFAR-10 training — SLURM job (single run, no chaining needed)
# Model: Spikformer-4-384 (4 layers, embed_dim=384, T=4)
# Expected runtime: ~2-4 hours on 8 H100s for 300 epochs
#
# Submission:
#   sbatch train_cifar10.sh
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=spikformer_cifar10
#SBATCH --partition=normal
#SBATCH --nodelist=multigpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1             # single task — torchrun spawns 8 processes internally
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=8:00:00
#SBATCH --output=/home/fritzsche/cifar10/logs/spikformer_cifar10_%j.out
#SBATCH --error=/home/fritzsche/cifar10/logs/spikformer_cifar10_%j.err

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR="/home/fritzsche/Spikformer/cifar10"
DATA_DIR="/home/fritzsche/cifar10"
OUTPUT_DIR="/nfsscratch/fritzsche/spikformer_output/cifar10"
LOG_DIR="/home/fritzsche/cifar10/logs"

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
LATEST_CKPT=$(ls -t "$OUTPUT_DIR"/spikformer_cifar10/last.pth.tar 2>/dev/null | head -1 || true)
if [[ -n "$LATEST_CKPT" ]]; then
    echo "Resuming from checkpoint: $LATEST_CKPT"
    RESUME_ARG="--resume $LATEST_CKPT"
else
    echo "No checkpoint found — starting fresh training."
fi

# ── Training ──────────────────────────────────────────────────────────────────
cd "$PROJECT_DIR"

torchrun \
    --standalone \
    --nproc_per_node=8 \
    train.py \
    --config cifar10.yml \
    -data-dir "$DATA_DIR" \
    --output "$OUTPUT_DIR" \
    --experiment spikformer_cifar10 \
    --batch-size 16 \
    --val-batch-size 32 \
    --workers 8 \
    --pin-mem \
    --checkpoint-hist 3 \
    --log-interval 100 \
    --seed 42 \
    $RESUME_ARG

echo "End: $(date)"
