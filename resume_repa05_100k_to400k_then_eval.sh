#!/bin/bash
# Resume the REPA-0.5 AttnScaf run from 100K to 400K on GPUs 2--5.
# After training, evaluate 200K on GPUs 2--3 and 400K on GPUs 4--5.
set -euo pipefail

CONFIG="configs/sit-xl2-attnscaf-dinob12-s4-t30k-cons-attn2-smooth-attn-repa8-0p5-norm-zs0p6-rproj-mlp.yaml"
EXP_NAME="sit-xl2-attnscaf-dinob12-s4-t30k-cons-attn2-smooth-attn-repa8-0p5-norm-zs0p6-rproj-mlp"
LOG_DIR="logs/${EXP_NAME}"

mkdir -p "$LOG_DIR"

echo "Resuming ${EXP_NAME} from 100K to 400K on GPUs 2,3,4,5"
./launch.sh \
    --config "$CONFIG" \
    --exp-name "$EXP_NAME" \
    --gpu 2,3,4,5 \
    --num-gpus 4 \
    --resume-step 100000 \
    --skip-eval

echo "Training complete. Starting parallel 200K and 400K evaluations."

./eval_ckpts.sh \
    --config "$CONFIG" \
    --exp-name "$EXP_NAME" \
    --steps "0200000" \
    --gpu 2,3 \
    --num-gpus 2 \
    --vae ema \
    > "${LOG_DIR}/eval_200k.log" 2>&1 &
PID_200K=$!

./eval_ckpts.sh \
    --config "$CONFIG" \
    --exp-name "$EXP_NAME" \
    --steps "0400000" \
    --gpu 4,5 \
    --num-gpus 2 \
    --vae ema \
    > "${LOG_DIR}/eval_400k.log" 2>&1 &
PID_400K=$!

echo "200K evaluation PID: ${PID_200K}"
echo "400K evaluation PID: ${PID_400K}"

STATUS_200K=0
STATUS_400K=0
wait "$PID_200K" || STATUS_200K=$?
wait "$PID_400K" || STATUS_400K=$?

echo "200K evaluation exit code: ${STATUS_200K}"
echo "400K evaluation exit code: ${STATUS_400K}"

if [[ "$STATUS_200K" -ne 0 || "$STATUS_400K" -ne 0 ]]; then
    echo "At least one evaluation failed. Check ${LOG_DIR}." >&2
    exit 1
fi

echo "Both evaluations completed successfully."
