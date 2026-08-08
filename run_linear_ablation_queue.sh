#!/usr/bin/env bash
set -euo pipefail

EXP1="sit-xl2-attnscaf-dinob12-s8-t30k-cons-kv2-smooth-attn-linear-repa-none"
EXP2="sit-xl2-attnscaf-dinob12-s4-t30k-cons-kv0p25-smooth-attn-linear-repa8-0p5-norm-none-rproj-mlp"
EXP3="sit-xl2-attnscaf-dinob12-s8-t30k-cons-kv2-smooth-none-repa-none"
STEP="0100000"
LOG_DIR="logs/linear_ablation_queue"

mkdir -p "$LOG_DIR"

echo "[Phase 1] Training EXP1 on GPUs 0-3 and EXP2 on GPUs 4-7"
./launch.sh \
  --config "configs/${EXP1}.yaml" \
  --exp-name "$EXP1" \
  --gpu 0,1,2,3 \
  --num-gpus 4 \
  --skip-eval \
  >"${LOG_DIR}/${EXP1}.train.log" 2>&1 &
pid_train1=$!

./launch.sh \
  --config "configs/${EXP2}.yaml" \
  --exp-name "$EXP2" \
  --gpu 4,5,6,7 \
  --num-gpus 4 \
  --skip-eval \
  >"${LOG_DIR}/${EXP2}.train.log" 2>&1 &
pid_train2=$!

phase1_failed=0
wait "$pid_train1" || phase1_failed=1
wait "$pid_train2" || phase1_failed=1
if [[ "$phase1_failed" -ne 0 ]]; then
  echo "Phase 1 failed. Check ${LOG_DIR}/*.train.log" >&2
  exit 1
fi

echo "[Phase 2] Evaluating EXP1 and EXP2 with 2 GPUs each; training EXP3 on GPUs 4-7"
./eval_ckpts.sh \
  --config "configs/${EXP1}.yaml" \
  --exp-name "$EXP1" \
  --steps "$STEP" \
  --gpu 0,1 \
  --num-gpus 2 \
  >"${LOG_DIR}/${EXP1}.eval.log" 2>&1 &
pid_eval1=$!

./eval_ckpts.sh \
  --config "configs/${EXP2}.yaml" \
  --exp-name "$EXP2" \
  --steps "$STEP" \
  --gpu 2,3 \
  --num-gpus 2 \
  >"${LOG_DIR}/${EXP2}.eval.log" 2>&1 &
pid_eval2=$!

./launch.sh \
  --config "configs/${EXP3}.yaml" \
  --exp-name "$EXP3" \
  --gpu 4,5,6,7 \
  --num-gpus 4 \
  --skip-eval \
  >"${LOG_DIR}/${EXP3}.train.log" 2>&1 &
pid_train3=$!

phase2_failed=0
wait "$pid_eval1" || phase2_failed=1
wait "$pid_eval2" || phase2_failed=1
wait "$pid_train3" || phase2_failed=1
if [[ "$phase2_failed" -ne 0 ]]; then
  echo "Phase 2 failed. Check ${LOG_DIR} logs." >&2
  exit 1
fi

echo "Queue complete. EXP1/EXP2 were evaluated at 100K; EXP3 finished training without evaluation."
