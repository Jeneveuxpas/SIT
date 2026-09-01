#!/usr/bin/env bash
# Train and evaluate the paired-x_0 K/V oracle.  Its FID is diagnostic only:
# clean reference information remains available at every sampling step.
set -euo pipefail

EXP_NAME="sit-xl2-attnscaf-src-latent-s8-t30k-oracle"

# Force the train+evaluate path even if a shell/session exported EVAL_ONLY.
EVAL_ONLY=false bash ./run_oracle_experiment.sh \
  --config "configs/${EXP_NAME}.yaml" \
  --exp-name "$EXP_NAME" \
  --step 0030000 \
  --gpu "${GPU:-0,1}" \
  --num-gpus "${NUM_GPUS:-2}" \
  --eval-batch-size "${EVAL_BATCH_SIZE:-128}" \
  --reference-pairing correct \
  --vae mse \
  --inference-dtype fp32
