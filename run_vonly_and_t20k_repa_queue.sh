#!/usr/bin/env bash
set -euo pipefail

V_EXP="sit-xl2-attnscaf-dinob12-replace-v-s8-t20k-cons-kv2-smooth-none-repa-none"
COMBINED_EXP="sit-xl2-attnscaf-dinob12-s4-t20k-cons-kv0p25-smooth-none-repa8-0p5-norm-none-rproj-mlp"
EVAL_STEP="0100000"
EVAL_DELAY_SECONDS=300
LOG_DIR="logs/vonly_t20k_repa_queue"

mkdir -p "$LOG_DIR"

echo "[1/3] Training V-only on GPUs 0-3 and the 400K combined run on GPUs 4-7."
bash ./launch.sh \
  --config "configs/${V_EXP}.yaml" \
  --exp-name "$V_EXP" \
  --gpu 0,1,2,3 \
  --num-gpus 4 \
  --skip-eval \
  >"${LOG_DIR}/${V_EXP}.train.log" 2>&1 &
v_train_pid=$!

bash ./launch.sh \
  --config "configs/${COMBINED_EXP}.yaml" \
  --exp-name "$COMBINED_EXP" \
  --gpu 4,5,6,7 \
  --num-gpus 4 \
  --skip-eval \
  >"${LOG_DIR}/${COMBINED_EXP}.train.log" 2>&1 &
combined_train_pid=$!

if ! wait "$v_train_pid"; then
  echo "V-only training failed. See ${LOG_DIR}/${V_EXP}.train.log" >&2
  exit 1
fi

echo "[2/3] V-only finished. Waiting for both 100K checkpoints."
v_ckpt="exps/${V_EXP}/checkpoints/${EVAL_STEP}.pt"
combined_ckpt="exps/${COMBINED_EXP}/checkpoints/${EVAL_STEP}.pt"
while [[ ! -f "$v_ckpt" || ! -f "$combined_ckpt" ]]; do
  if ! kill -0 "$combined_train_pid" 2>/dev/null; then
    echo "Combined training stopped before both 100K checkpoints were available." >&2
    wait "$combined_train_pid" || true
    exit 1
  fi
  sleep 30
done

echo "Both 100K checkpoints found. Waiting ${EVAL_DELAY_SECONDS}s before evaluation."
sleep "$EVAL_DELAY_SECONDS"

echo "[3/3] Evaluating both 100K checkpoints on GPUs 0-3 while combined training continues."
bash ./eval_ckpts.sh \
  --config "configs/${V_EXP}.yaml" \
  --exp-name "$V_EXP" \
  --steps "$EVAL_STEP" \
  --gpu 0,1 \
  --num-gpus 2 \
  >"${LOG_DIR}/${V_EXP}.eval100k.log" 2>&1 &
v_eval_pid=$!

bash ./eval_ckpts.sh \
  --config "configs/${COMBINED_EXP}.yaml" \
  --exp-name "$COMBINED_EXP" \
  --steps "$EVAL_STEP" \
  --gpu 2,3 \
  --num-gpus 2 \
  >"${LOG_DIR}/${COMBINED_EXP}.eval100k.log" 2>&1 &
combined_eval_pid=$!

status=0
wait "$v_eval_pid" || status=1
wait "$combined_eval_pid" || status=1
wait "$combined_train_pid" || status=1

if [[ "$status" -ne 0 ]]; then
  echo "At least one job failed. Check ${LOG_DIR}." >&2
  exit 1
fi

echo "Queue complete: both 100K evaluations finished and the combined run reached 400K."
