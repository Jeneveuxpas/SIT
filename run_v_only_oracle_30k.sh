#!/usr/bin/env bash
set -euo pipefail

EXP_NAME="sit-xl2-attnscaf-dinob12-replace-v-s8-t30k-cons-none-smooth-none-repa-none"

bash ./run_oracle_experiment.sh \
  --config "configs/${EXP_NAME}.yaml" \
  --exp-name "$EXP_NAME" \
  --step 0030000 \
  --gpu "${GPU:-3}" \
  --num-gpus 1 \
  --eval-batch-size "${EVAL_BATCH_SIZE:-256}" \
  --reference-pairing correct \
  --vae ema \
  --inference-dtype fp32
