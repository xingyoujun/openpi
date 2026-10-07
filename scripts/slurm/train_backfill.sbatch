#!/bin/bash
# Auto-resuming openpi training for the (preemptible) backfill partition.
#
#   sbatch scripts/slurm/train_backfill.sbatch <config-name> <exp-name> [extra train.py args...]
#   e.g.   sbatch scripts/slurm/train_backfill.sbatch pi05_pine_3task pine3 --num-train-steps=20000
#
# How it survives preemption / the time limit:
#   * --requeue            : Slurm puts the job back in the queue when it is preempted.
#   * SIGTERM (preemption) and SIGUSR1 (--signal, 10 min before the time limit) are forwarded to train.py, which
#     saves a checkpoint at the current step and exits with code 85.
#   * exit 85 -> this script calls `scontrol requeue`; any restart finds the checkpoint dir and passes --resume.
#   * only the 2 newest checkpoints are kept (--max-to-keep=2, --keep-period=None).
#   * periodic saves are rare (SAVE_INTERVAL, default 5000 steps); the signal-triggered save is the main safety net.
#
# Optional environment variables (pass with `sbatch --export=ALL,NAME=value ...`):
#   SAVE_INTERVAL (5000)  NUM_WORKERS (16)  WANDB_MODE (online if a key is found, else offline)  IMAGE  MAX_RESTARTS
#   wandb key: put it in $LUSTRE/.wandb_api_key (chmod 600) or export WANDB_API_KEY.
#SBATCH -A coreai_devtech_all
#SBATCH -p backfill
#SBATCH --qos=normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -t 7-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --open-mode=append
#SBATCH -J openpi-train
#SBATCH -o /lustre/fsw/portfolios/coreai/users/chuanruiz/logs/%x_%j.out

CONFIG=${1:?usage: sbatch train_backfill.sbatch <config-name> <exp-name> [train.py args...]}
EXP=${2:?usage: sbatch train_backfill.sbatch <config-name> <exp-name> [train.py args...]}
shift 2
EXTRA_ARGS="$*"

LUSTRE=/lustre/fsw/portfolios/coreai/users/chuanruiz
OPENPI=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/chuanruiz/code/openpi
IMAGE=${IMAGE:-$LUSTRE/sqsh/dev.sqsh}
SAVE_INTERVAL=${SAVE_INTERVAL:-5000}
NUM_WORKERS=${NUM_WORKERS:-16}
MAX_RESTARTS=${MAX_RESTARTS:-200}      # safety net against requeue loops
PREEMPT_EXIT_CODE=85                   # must match scripts/train.py
CKPT_DIR=$OPENPI/checkpoints/$CONFIG/$EXP

RESTARTS=${SLURM_RESTART_COUNT:-0}
echo "[sbatch] job=$SLURM_JOB_ID restart=$RESTARTS node=$(hostname) config=$CONFIG exp=$EXP $(date)"

# Resume if a previous run left a checkpoint directory, else start fresh.
if [ -d "$CKPT_DIR" ]; then RESUME_FLAG="--resume"; else RESUME_FLAG=""; fi
echo "[sbatch] resume flag: '${RESUME_FLAG}'"

srun --container-image="$IMAGE" \
     --container-mounts=/lustre:/lustre,/home/chuanruiz:/home/chuanruiz \
     --no-container-mount-home \
     --export=ALL \
     bash -c "
       set -e
       source /home/chuanruiz/env.sh            # also cd's to \$CODE_DIR; we cd again below
       command -v ffmpeg >/dev/null || (apt-get install -y -q ffmpeg >/dev/null 2>&1 || true)
       cd $OPENPI
       export HF_LEROBOT_HOME=$LUSTRE/data
       export OPENPI_DATA_HOME=$LUSTRE/cache/openpi
       export UV_PYTHON_INSTALL_DIR=$LUSTRE/uv-python
       export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
       if [ -z \"\${WANDB_API_KEY:-}\" ] && [ -f $LUSTRE/.wandb_api_key ]; then export WANDB_API_KEY=\$(cat $LUSTRE/.wandb_api_key); fi
       if [ -n \"\${WANDB_API_KEY:-}\" ]; then export WANDB_MODE=\${WANDB_MODE:-online}; else export WANDB_MODE=\${WANDB_MODE:-offline}; fi
       echo \"[train] WANDB_MODE=\$WANDB_MODE\"
       exec .venv/bin/python scripts/train.py $CONFIG --exp-name=$EXP $RESUME_FLAG \
            --max-to-keep=2 --keep-period=None --save-interval=$SAVE_INTERVAL --num-workers=$NUM_WORKERS $EXTRA_ARGS
     " &
SRUN_PID=$!

# Forward preemption (SIGTERM) and the pre-time-limit warning (SIGUSR1) to the training process.
forward() {
  echo "[sbatch] caught signal at $(date); forwarding SIGUSR1 to training"
  kill -USR1 "$SRUN_PID" 2>/dev/null
}
trap forward USR1 TERM

# `wait` returns early (>128) when a trapped signal arrives, so wait until srun is really gone.
while true; do
  wait "$SRUN_PID"; RC=$?
  kill -0 "$SRUN_PID" 2>/dev/null || break
done
echo "[sbatch] training exited rc=$RC at $(date)"

if [ "$RC" -eq "$PREEMPT_EXIT_CODE" ]; then
  if [ "$RESTARTS" -ge "$MAX_RESTARTS" ]; then
    echo "[sbatch] reached MAX_RESTARTS=$MAX_RESTARTS, not requeueing"; exit 1
  fi
  echo "[sbatch] checkpoint saved; requeueing job $SLURM_JOB_ID"
  scontrol requeue "$SLURM_JOB_ID" || echo "[sbatch] requeue refused (probably already being requeued by Slurm)"
  exit 0
fi
exit "$RC"
