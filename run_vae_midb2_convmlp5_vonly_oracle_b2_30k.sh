#!/usr/bin/env bash
# Train and evaluate the paired VAE mid.block_2 V-only oracle.
set -euo pipefail

EXP_NAME="sit-b2-attnscaf2-vaemidb2-convmlp5-s6-t30k-vonly-oracle"
CONFIG="configs/${EXP_NAME}.yaml"
GPU_LIST="${GPU:-0,1}"
WORLD_SIZE="${NUM_GPUS:-2}"
FID_SAMPLES="${NUM_FID_SAMPLES:-10000}"
MASTER_PORT=$((29500 + RANDOM % 1000))
CHECKPOINT="exps/${EXP_NAME}/checkpoints/0030000.pt"

export CUDA_VISIBLE_DEVICES="$GPU_LIST"
if [[ -f "$CHECKPOINT" && "${FORCE_TRAIN:-false}" != "true" ]]; then
  echo "Using existing checkpoint: ${CHECKPOINT}"
else
  echo "Training VAE mid.block_2 conv+MLP5 V-only oracle: ${EXP_NAME} (${WORLD_SIZE} GPU(s))"
  if [[ "$WORLD_SIZE" -gt 1 ]]; then
    accelerate launch \
      --multi_gpu \
      --main_process_port "$MASTER_PORT" \
      --num_processes "$WORLD_SIZE" \
      train.py --exp-name "$EXP_NAME" --seed 0 --config "$CONFIG"
  else
    accelerate launch \
      --main_process_port "$MASTER_PORT" \
      --num_processes 1 \
      train.py --exp-name "$EXP_NAME" --seed 0 --config "$CONFIG"
  fi
fi

bash ./run_oracle_experiment.sh \
  --config "$CONFIG" \
  --exp-name "$EXP_NAME" \
  --step 0030000 \
  --gpu "$GPU_LIST" \
  --num-gpus "$WORLD_SIZE" \
  --num-fid-samples "$FID_SAMPLES" \
  --eval-batch-size "${EVAL_BATCH_SIZE:-64}" \
  --reference-pairing correct \
  --vae mse \
  --inference-dtype fp32 \
  --eval-only
