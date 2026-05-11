#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Spikformer ImageNet training — SLURM job with automatic chaining
#
# Each job runs for up to WALL_TIME hours, then resubmits itself until all
# TARGET_EPOCHS epochs are finished.
#
# First submission:
#   sbatch train_imagenet.slurm
#
# Subsequent jobs are chained automatically via sbatch --dependency.
# You can also re-submit manually at any time — it will resume from the latest
# checkpoint found in OUTPUT_DIR.
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=spikformer_imagenet
#SBATCH --partition=gpu                  # adjust to your cluster's GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4             # one task per GPU
#SBATCH --gres=gpu:4                    # 4 GPUs per node
#SBATCH --cpus-per-task=8              # 8 CPU workers per GPU
#SBATCH --mem=256G
#SBATCH --time=24:00:00                 # wall-clock limit
#SBATCH --output=/nfsscratch/fritzsche/logs/spikformer_%j.out
#SBATCH --error=/nfsscratch/fritzsche/logs/spikformer_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marcusfritzsche1@gmail.com

# ── User-configurable paths ───────────────────────────────────────────────────
CONDA_ENV="spikformer"                              # conda environment name
PROJECT_DIR="$HOME/projects/spikformer/imagenet"   # path to this repo's imagenet/ folder
DATA_DIR="/nfsscratch/fritzsche/imagenet"
OUTPUT_DIR="/nfsscratch/fritzsche/spikformer_output"
LOG_DIR="/nfsscratch/fritzsche/logs"

# Training hyperparameters (override imagenet.yml defaults here if needed)
TARGET_EPOCHS=310      # total epochs to train (paper: 310)
NUM_GPUS=4
BATCH_SIZE=32          # per-GPU batch size  (32 * 4 GPUs = 128 total, matching paper)
VAL_BATCH_SIZE=32
WORKERS=8              # dataloader workers per GPU
WALL_TIME=23           # stop gracefully this many hours into the job

# ── Environment setup ─────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

# Load required modules (adapt to your HPC's module system)
module purge
module load cuda/11.8          # adjust version as needed
module load cudnn/8.6-cuda-11.8
module load gcc/11.3

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# ── Checkpoint auto-detection ─────────────────────────────────────────────────
# timm's CheckpointSaver writes checkpoint-{epoch}.pth.tar; pick the newest.
RESUME_ARG=""
LATEST_CKPT=$(ls -t "$OUTPUT_DIR"/spikformer/checkpoint-*.pth.tar 2>/dev/null | head -1 || true)
if [[ -n "$LATEST_CKPT" ]]; then
    echo "Resuming from checkpoint: $LATEST_CKPT"
    RESUME_ARG="--resume $LATEST_CKPT"
else
    echo "No checkpoint found — starting fresh training."
fi

# ── Determine how many epochs we have already completed ──────────────────────
COMPLETED_EPOCHS=0
if [[ -n "$LATEST_CKPT" ]]; then
    COMPLETED_EPOCHS=$(python3 - <<EOF
import torch, sys
try:
    ckpt = torch.load("$LATEST_CKPT", map_location="cpu")
    print(ckpt.get("epoch", 0))
except Exception as e:
    print(0)
EOF
)
fi
echo "Epochs completed so far: $COMPLETED_EPOCHS / $TARGET_EPOCHS"

if [[ "$COMPLETED_EPOCHS" -ge "$TARGET_EPOCHS" ]]; then
    echo "Training already complete. Exiting."
    exit 0
fi

# ── Run training ──────────────────────────────────────────────────────────────
cd "$PROJECT_DIR"

torchrun \
    --standalone \
    --nproc_per_node="$NUM_GPUS" \
    train.py \
    --config imagenet.yml \
    -data-dir "$DATA_DIR" \
    --dataset imagenet \
    --train-split train \
    --val-split val \
    --model spikformer \
    --num-classes 1000 \
    --output "$OUTPUT_DIR" \
    --experiment spikformer \
    --epochs "$TARGET_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --val-batch-size "$VAL_BATCH_SIZE" \
    --workers "$WORKERS" \
    --opt adamw \
    --lr 5e-4 \
    --min-lr 1e-5 \
    --weight-decay 5e-2 \
    --sched cosine \
    --warmup-epochs 20 \
    --warmup-lr 1e-6 \
    --cooldown-epochs 10 \
    --aa rand-m9-mstd0.5-inc1 \
    --mixup 0.8 \
    --cutmix 1.0 \
    --mixup-prob 1.0 \
    --mixup-switch-prob 0.5 \
    --mixup-mode batch \
    --reprob 0.25 \
    --remode const \
    --smoothing 0.1 \
    --amp \
    --pin-mem \
    --dist-bn reduce \
    --sync-bn \
    --checkpoint-hist 3 \
    --log-interval 500 \
    --seed 42 \
    $RESUME_ARG

TRAIN_EXIT=$?

# ── Auto-resubmit if training is not yet complete ────────────────────────────
if [[ $TRAIN_EXIT -ne 0 ]]; then
    echo "Training exited with code $TRAIN_EXIT — not resubmitting."
    exit $TRAIN_EXIT
fi

# Check completed epoch count again after this job finished
NEW_LATEST=$(ls -t "$OUTPUT_DIR"/spikformer/checkpoint-*.pth.tar 2>/dev/null | head -1 || true)
NEW_COMPLETED=0
if [[ -n "$NEW_LATEST" ]]; then
    NEW_COMPLETED=$(python3 - <<EOF
import torch, sys
try:
    ckpt = torch.load("$NEW_LATEST", map_location="cpu")
    print(ckpt.get("epoch", 0))
except Exception:
    print(0)
EOF
)
fi

echo "Epochs completed after this job: $NEW_COMPLETED / $TARGET_EPOCHS"

if [[ "$NEW_COMPLETED" -lt "$TARGET_EPOCHS" ]]; then
    echo "Submitting next chained job ..."
    NEXT_JOB=$(sbatch --parsable "$0")
    echo "  Next job ID: $NEXT_JOB"
else
    echo "Training complete! All $TARGET_EPOCHS epochs finished."
    echo "Best checkpoint: $OUTPUT_DIR/spikformer/model_best.pth.tar"
fi
