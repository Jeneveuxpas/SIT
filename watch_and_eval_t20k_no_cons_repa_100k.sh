#!/usr/bin/env bash
set -euo pipefail

EXP_NAME="sit-xl2-attnscaf-dinob12-s4-t20k-cons-none-smooth-none-repa8-0p5-norm-none-rproj-mlp"
CONFIG="configs/${EXP_NAME}.yaml"
CKPT_DIR="exps/${EXP_NAME}/checkpoints"
EVAL_STEP="0100000"
CKPT_FILE="${CKPT_DIR}/${EVAL_STEP}.pt"
POLL_SECONDS=30
EVAL_DELAY_SECONDS=300
LOG_DIR="logs/${EXP_NAME}_eval_100k"

mkdir -p "$LOG_DIR"

echo "Waiting for the 100K checkpoint of ${EXP_NAME}."
while [[ ! -f "$CKPT_FILE" ]]; do
  sleep "$POLL_SECONDS"
done

# Require the checkpoint size to remain unchanged for one polling window.
previous_size=-1
while true; do
  current_size=$(stat -c %s "$CKPT_FILE" 2>/dev/null || stat -f %z "$CKPT_FILE")
  if [[ "$current_size" -gt 0 && "$current_size" -eq "$previous_size" ]]; then
    break
  fi
  previous_size="$current_size"
  sleep "$POLL_SECONDS"
done

echo "The 100K checkpoint is stable. Waiting ${EVAL_DELAY_SECONDS}s before evaluation."
sleep "$EVAL_DELAY_SECONDS"

echo "Evaluating 100K on GPUs 0,1."
bash ./eval_ckpts.sh \
  --config "$CONFIG" \
  --exp-name "$EXP_NAME" \
  --steps "$EVAL_STEP" \
  --gpu 0,1 \
  --num-gpus 2 \
  >"${LOG_DIR}/eval_100k.log" 2>&1

echo "The 100K evaluation is complete. Log: ${LOG_DIR}/eval_100k.log"
