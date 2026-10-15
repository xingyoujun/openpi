#!/bin/bash
# [cw-dfw-cs-001] Submit pi05_pine_usdcraft3_fast: 1 node x 8 H100, 30k steps, auto-resuming (see train_backfill.sbatch).
#
#   bash scripts/slurm/submit_usdcraft3.sh [exp-name] [extra train.py args...]
#
# Queues on batch and backfill together with a 4h limit (batch 8-GPU waits are ~0h, backfill ~6h); every segment
# saves a checkpoint before the limit and requeues itself, the next one resumes.
set -e
EXP=${1:-usdcraft3}
shift || true
cd /lustre/fsw/portfolios/coreai/users/chuanruiz/code/openpi
sbatch -p batch,backfill -t 4:00:00 --gpus-per-node=8 -J openpi-usdcraft3 \
  --export=ALL,NUM_WORKERS=48,SAVE_INTERVAL=5000 \
  scripts/slurm/train_backfill.sbatch pi05_pine_usdcraft3_fast "$EXP" \
  --batch-size=64 --num-train-steps=30000 "$@"
