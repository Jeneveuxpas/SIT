#!/usr/bin/env bash
# Train and evaluate the paired-x_0 K/V oracle.  Its FID is diagnostic only:
# clean reference information remains available at every sampling step.
set -euo pipefail

EXP_NAME="sit-xl2-attnscaf-src-latent-s8-t30k-oracle"
CONFIG="configs/${EXP_NAME}.yaml"
GPU_LIST="${GPU:-0,1}"
WORLD_SIZE="${NUM_GPUS:-2}"
MASTER_PORT=$((29500 + RANDOM % 1000))

# Keep the train step here rather than relying on run_oracle_experiment.sh:
# some remote copies of that general-purpose script are evaluation-only.
export CUDA_VISIBLE_DEVICES="$GPU_LIST"
echo "Training paired-x_0 K/V oracle: ${EXP_NAME} (${WORLD_SIZE} GPU(s))"
if [[ "$WORLD_SIZE" -gt 1 ]]; then
  accelerate launch \
    --multi_gpu \
    --main_process_port "$MASTER_PORT" \
    --num_processes "$WORLD_SIZE" \
    train.py \
    --exp-name "$EXP_NAME" \
    --seed 0 \
    --config "$CONFIG"
else
  accelerate launch \
    --main_process_port "$MASTER_PORT" \
    --num_processes 1 \
    train.py \
    --exp-name "$EXP_NAME" \
    --seed 0 \
    --config "$CONFIG"
fi

# The existing oracle driver now only needs to find the checkpoint and evaluate it.
bash ./run_oracle_experiment.sh \
  --config "$CONFIG" \
  --exp-name "$EXP_NAME" \
  --step 0030000 \
  --gpu "$GPU_LIST" \
  --num-gpus "$WORLD_SIZE" \
  --eval-batch-size "${EVAL_BATCH_SIZE:-128}" \
  --reference-pairing correct \
  --vae mse \
  --inference-dtype fp32
