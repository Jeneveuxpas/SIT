#!/usr/bin/env bash
# Train the normal (unpaired) DINOv2-B layer-12 K/V-replacement baseline.
set -euo pipefail

EXP_NAME="sit-b2-attnscaf-dinob12-s6-t30k-kvreplace-baseline"
CONFIG="configs/${EXP_NAME}.yaml"
GPU_LIST="${GPU:-0,1}"
WORLD_SIZE="${NUM_GPUS:-2}"
FID_SAMPLES="${NUM_FID_SAMPLES:-10000}"
EVAL_VAE="${VAE:-ema}"
CHECKPOINT="exps/${EXP_NAME}/checkpoints/0030000.pt"

if [[ -f "$CHECKPOINT" && "${FORCE_TRAIN:-false}" != "true" ]]; then
  echo "Using existing checkpoint: ${CHECKPOINT}"
else
  GPU="$GPU_LIST" NUM_GPUS="$WORLD_SIZE" \
    bash ./launch.sh \
      --config "$CONFIG" \
      --exp-name "$EXP_NAME" \
      --gpu "$GPU_LIST" \
      --num-gpus "$WORLD_SIZE" \
      --skip-eval
fi

GPU="$GPU_LIST" NUM_GPUS="$WORLD_SIZE" NUM_FID_SAMPLES="$FID_SAMPLES" VAE="$EVAL_VAE" \
  bash ./launch.sh \
    --config "$CONFIG" \
    --exp-name "$EXP_NAME" \
    --gpu "$GPU_LIST" \
    --num-gpus "$WORLD_SIZE" \
    --eval-only \
    --eval-steps 0030000
