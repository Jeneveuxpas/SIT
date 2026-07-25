#!/bin/bash
# Train one oracle-interface experiment, generate reference-conditioned samples,
# and calculate FID. This script does not call launch.sh or generate.py.
set -euo pipefail

CONFIG=""
EXP_NAME=""
GPU="0"
NUM_GPUS="1"
SEED="0"
STEP="0030000"
DATA_DIR="/dev/shm/data"
REFERENCE_DIR=""
REF_BATCH="/workspace/SIT/VIRTUAL_imagenet256_labeled.npz"
NUM_FID_SAMPLES="50000"
EVAL_BATCH_SIZE="16"
EVAL_NUM_STEPS="250"
CFG_SCALE="1.0"
MODE="sde"
VAE="ema"
INFERENCE_DTYPE="fp32"
EVAL_ONLY="false"
REFERENCE_PAIRING="correct"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --exp-name) EXP_NAME="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --num-gpus) NUM_GPUS="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --step) STEP="$2"; shift 2 ;;
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        --reference-dir) REFERENCE_DIR="$2"; shift 2 ;;
        --reference-pairing) REFERENCE_PAIRING="$2"; shift 2 ;;
        --ref-batch) REF_BATCH="$2"; shift 2 ;;
        --num-fid-samples) NUM_FID_SAMPLES="$2"; shift 2 ;;
        --eval-batch-size) EVAL_BATCH_SIZE="$2"; shift 2 ;;
        --eval-num-steps) EVAL_NUM_STEPS="$2"; shift 2 ;;
        --cfg-scale) CFG_SCALE="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --vae) VAE="$2"; shift 2 ;;
        --inference-dtype) INFERENCE_DTYPE="$2"; shift 2 ;;
        --eval-only) EVAL_ONLY="true"; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$CONFIG" || -z "$EXP_NAME" ]]; then
    echo "Usage: $0 --config CONFIG --exp-name NAME --gpu GPU" >&2
    exit 1
fi
if [[ "$NUM_GPUS" -lt 1 ]]; then
    echo "--num-gpus must be at least 1." >&2
    exit 1
fi
if [[ -z "$REFERENCE_DIR" ]]; then
    REFERENCE_DIR="${DATA_DIR}/imagenet-latents-images/val"
fi

export CUDA_VISIBLE_DEVICES="$GPU"
MASTER_PORT=$((29500 + RANDOM % 1000))
SAVE_PATH="exps/${EXP_NAME}"
CHECKPOINT="${SAVE_PATH}/checkpoints/${STEP}.pt"
ORACLE_DIR="${SAVE_PATH}/checkpoints/${EXP_NAME}_oracle-${REFERENCE_PAIRING}-vae${VAE}-cfg${CFG_SCALE}-seed${SEED}-mode${MODE}-steps${EVAL_NUM_STEPS}_${STEP}"
SAMPLE_NPZ="${ORACLE_DIR}.npz"

if [[ "$EVAL_ONLY" == "false" ]]; then
    echo "Training oracle experiment: ${EXP_NAME}"
    if [[ "$NUM_GPUS" -gt 1 ]]; then
        accelerate launch \
            --multi_gpu \
            --main_process_port "$MASTER_PORT" \
            --num_processes "$NUM_GPUS" \
            train.py \
            --exp-name "$EXP_NAME" \
            --seed "$SEED" \
            --config "$CONFIG"
    else
        accelerate launch \
            --main_process_port "$MASTER_PORT" \
            --num_processes 1 \
            train.py \
            --exp-name "$EXP_NAME" \
            --seed "$SEED" \
            --config "$CONFIG"
    fi
else
    echo "Skipping training and evaluating existing checkpoint: ${CHECKPOINT}"
fi

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi

echo "Generating ${NUM_FID_SAMPLES} reference-conditioned oracle samples"
torchrun \
    --nproc_per_node="$NUM_GPUS" \
    --master_port="$MASTER_PORT" \
    scripts/generate_oracle_fid.py \
    --checkpoint "$CHECKPOINT" \
    --reference-dir "$REFERENCE_DIR" \
    --reference-pairing "$REFERENCE_PAIRING" \
    --output-dir "$ORACLE_DIR" \
    --output-npz "$SAMPLE_NPZ" \
    --num-fid-samples "$NUM_FID_SAMPLES" \
    --batch-size "$EVAL_BATCH_SIZE" \
    --global-seed "$SEED" \
    --mode "$MODE" \
    --num-steps "$EVAL_NUM_STEPS" \
    --cfg-scale "$CFG_SCALE" \
    --vae "$VAE" \
    --inference-dtype "$INFERENCE_DTYPE"

echo "Calculating oracle FID"
python evaluations/evaluator.py \
    --ref_batch "$REF_BATCH" \
    --sample_batch "$SAMPLE_NPZ" \
    --save_path "${SAVE_PATH}/checkpoints" \
    --step "$STEP" \
    --num_steps "$EVAL_NUM_STEPS" \
    --cfg "$CFG_SCALE"

echo "Oracle experiment complete: ${EXP_NAME}"
