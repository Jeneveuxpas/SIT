#!/usr/bin/env bash
set -euo pipefail

EXP_NAME="sit-xl2-attnscaf-dinob12-s4-t20k-cons-kv0p25-smooth-none-repa8-0p5-norm-none-rproj-mlp"
CONFIG="configs/${EXP_NAME}.yaml"
CKPT_DIR="exps/${EXP_NAME}/checkpoints"
CKPT_200K="${CKPT_DIR}/0200000.pt"
CKPT_400K="${CKPT_DIR}/0400000.pt"
POLL_SECONDS=30
EVAL_DELAY_SECONDS=300
LOG_DIR="logs/${EXP_NAME}_eval_200k_400k"

mkdir -p "$LOG_DIR"

echo "Waiting for the 200K and 400K checkpoints of ${EXP_NAME}."
while [[ ! -f "$CKPT_200K" || ! -f "$CKPT_400K" ]]; do
  sleep "$POLL_SECONDS"
done

# A checkpoint path may become visible while torch.save is still writing it.
# Require the final checkpoint size to remain unchanged for one polling window.
previous_size=-1
while true; do
  current_size=$(stat -c %s "$CKPT_400K" 2>/dev/null || stat -f %z "$CKPT_400K")
  if [[ "$current_size" -gt 0 && "$current_size" -eq "$previous_size" ]]; then
    break
  fi
  previous_size="$current_size"
  sleep "$POLL_SECONDS"
done

echo "Both checkpoints are stable. Waiting ${EVAL_DELAY_SECONDS}s before evaluation."
sleep "$EVAL_DELAY_SECONDS"

echo "Evaluating 200K on GPUs 4,5 and 400K on GPUs 6,7."
bash ./eval_ckpts.sh \
  --config "$CONFIG" \
  --exp-name "$EXP_NAME" \
  --steps "0200000" \
  --gpu 4,5 \
  --num-gpus 2 \
  >"${LOG_DIR}/eval_200k.log" 2>&1 &
eval_200k_pid=$!

bash ./eval_ckpts.sh \
  --config "$CONFIG" \
  --exp-name "$EXP_NAME" \
  --steps "0400000" \
  --gpu 6,7 \
  --num-gpus 2 \
  >"${LOG_DIR}/eval_400k.log" 2>&1 &
eval_400k_pid=$!

status=0
wait "$eval_200k_pid" || status=1
wait "$eval_400k_pid" || status=1

if [[ "$status" -ne 0 ]]; then
  echo "At least one evaluation failed. Check ${LOG_DIR}." >&2
  exit 1
fi

echo "The 200K and 400K evaluations are complete. Logs: ${LOG_DIR}"
