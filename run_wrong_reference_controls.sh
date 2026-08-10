#!/bin/bash
# Evaluate the existing 30K internal-K/V AttnScaf oracle checkpoint using
# same-class and different-class incorrect clean references on two GPUs.
set -euo pipefail

CONFIG="${CONFIG:-configs/sit-xl2-attnscaf-dinob12-s8-t30k-cons-none-smooth-none-repa-none.yaml}"
EXP_NAME="${EXP_NAME:-attnscaf-kv-layer8-oracle30k}"
GPU="${GPU:-0,1}"
NUM_GPUS="${NUM_GPUS:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"

echo "Running same-class wrong-reference control"
./run_oracle_experiment.sh \
    --config "$CONFIG" \
    --exp-name "$EXP_NAME" \
    --gpu "$GPU" \
    --num-gpus "$NUM_GPUS" \
    --eval-only \
    --eval-batch-size "$EVAL_BATCH_SIZE" \
    --reference-pairing same_class_wrong

echo "Running different-class wrong-reference control"
./run_oracle_experiment.sh \
    --config "$CONFIG" \
    --exp-name "$EXP_NAME" \
    --gpu "$GPU" \
    --num-gpus "$NUM_GPUS" \
    --eval-only \
    --eval-batch-size "$EVAL_BATCH_SIZE" \
    --reference-pairing different_class_wrong

echo "Wrong-reference controls completed."
