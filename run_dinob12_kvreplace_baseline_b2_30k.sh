#!/usr/bin/env bash
# Train and evaluate the paired-reference DINOv2-B layer-12 K/V oracle.
set -euo pipefail

EXP_NAME="sit-b2-attnscaf-dinob12-s6-t30k-kvreplace-baseline"
CONFIG="configs/${EXP_NAME}.yaml"
GPU_LIST="${GPU:-0,1}"
WORLD_SIZE="${NUM_GPUS:-2}"
FID_SAMPLES="${NUM_FID_SAMPLES:-10000}"
EVAL_VAE="${VAE:-mse}"
CHECKPOINT="exps/${EXP_NAME}/checkpoints/0030000.pt"

if [[ -f "$CHECKPOINT" && "${FORCE_TRAIN:-false}" != "true" ]]; then
  echo "Using existing checkpoint: ${CHECKPOINT}"
else
  export CUDA_VISIBLE_DEVICES="$GPU_LIST"
  MASTER_PORT=$((29500 + RANDOM % 1000))
  echo "Training paired DINO12 K/V oracle: ${EXP_NAME} (${WORLD_SIZE} GPU(s))"
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
  --gpu "$GPU_LIST" \
  --num-gpus "$WORLD_SIZE" \
  --step 0030000 \
  --num-fid-samples "$FID_SAMPLES" \
  --eval-batch-size "${EVAL_BATCH_SIZE:-64}" \
  --reference-pairing correct \
  --vae "$EVAL_VAE" \
  --inference-dtype fp32 \
  --eval-only
