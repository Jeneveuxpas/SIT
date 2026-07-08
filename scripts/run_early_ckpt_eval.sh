#!/usr/bin/env bash
set -euo pipefail

# Run early-checkpoint mechanism evaluations for an experiment.
#
# Default behavior:
#   - discovers all checkpoints in the run up to 100k
#   - runs q_scaffold_probe.py
#   - runs compute_spatial_metrics.py
#   - skips FID and linear probe unless requested
#
# Example:
#   CUDA_VISIBLE_DEVICES=0 ./scripts/run_early_ckpt_eval.sh \
#     --exp-name attnscaf-ablate-kvnorm-attnscaf-only-100k \
#     --data-dir /dev/shm/data \
#     --teacher-align --teacher-layer-depths 8,10 \
#     --steps 0005000,0010000,0015000,0020000,0025000,0030000,0035000,0040000,0050000,0075000,0100000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

EXP_NAME=""
RUN_DIR=""
CONFIG=""
STEPS=""
MAX_STEP=100000
DATA_DIR="/dev/shm/data"
GPU="${GPU:-0}"
DEVICE="${DEVICE:-cuda:0}"
SPLIT="train"
NUM_SAMPLES=512
BATCH_SIZE=64
NUM_WORKERS=12
TIMESTEPS="0.0,0.1,0.5"
LAYER_DEPTHS=""
INFERENCE_DTYPE="fp32"
PATCH_SHUFFLE_MODE="checkpoint"

SKIP_Q_PROBE="false"
SKIP_SPATIAL="false"
RUN_TEACHER_ALIGN="false"
TEACHER_LAYER_DEPTHS="8,10"
TEACHER_PATCH_SHUFFLE_MODE="off"
RUN_LINEAR_PROBE="false"
RUN_FID="false"
FID_STEPS=""
NUM_FID_SAMPLES="${NUM_FID_SAMPLES:-50000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
EVAL_NUM_STEPS="${EVAL_NUM_STEPS:-250}"
CFG_SCALE="${CFG_SCALE:-1.0}"
GUIDANCE_LOW="${GUIDANCE_LOW:-0.0}"
GUIDANCE_HIGH="${GUIDANCE_HIGH:-1.0}"
VAE="${VAE:-mse}"
REF_BATCH="${REF_BATCH:-/workspace/SIT/VIRTUAL_imagenet256_labeled.npz}"

usage() {
    sed -n '1,36p' "$0"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --exp-name)
            EXP_NAME="$2"
            shift 2
            ;;
        --run-dir)
            RUN_DIR="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --max-step)
            MAX_STEP="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --timesteps)
            TIMESTEPS="$2"
            shift 2
            ;;
        --layer-depths)
            LAYER_DEPTHS="$2"
            shift 2
            ;;
        --inference-dtype)
            INFERENCE_DTYPE="$2"
            shift 2
            ;;
        --patch-shuffle-mode)
            PATCH_SHUFFLE_MODE="$2"
            shift 2
            ;;
        --skip-q-probe)
            SKIP_Q_PROBE="true"
            shift
            ;;
        --skip-spatial)
            SKIP_SPATIAL="true"
            shift
            ;;
        --teacher-align)
            RUN_TEACHER_ALIGN="true"
            shift
            ;;
        --teacher-layer-depths)
            TEACHER_LAYER_DEPTHS="$2"
            shift 2
            ;;
        --teacher-patch-shuffle-mode)
            TEACHER_PATCH_SHUFFLE_MODE="$2"
            shift 2
            ;;
        --linear-probe)
            RUN_LINEAR_PROBE="true"
            shift
            ;;
        --fid)
            RUN_FID="true"
            shift
            ;;
        --fid-steps)
            FID_STEPS="$2"
            shift 2
            ;;
        --num-fid-samples)
            NUM_FID_SAMPLES="$2"
            shift 2
            ;;
        --eval-batch-size)
            EVAL_BATCH_SIZE="$2"
            shift 2
            ;;
        --eval-num-steps)
            EVAL_NUM_STEPS="$2"
            shift 2
            ;;
        --cfg-scale)
            CFG_SCALE="$2"
            shift 2
            ;;
        --guidance-low)
            GUIDANCE_LOW="$2"
            shift 2
            ;;
        --guidance-high)
            GUIDANCE_HIGH="$2"
            shift 2
            ;;
        --vae)
            VAE="$2"
            shift 2
            ;;
        --ref-batch)
            REF_BATCH="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ -z "$RUN_DIR" ]]; then
    if [[ -z "$EXP_NAME" ]]; then
        echo "Please pass --exp-name or --run-dir" >&2
        exit 1
    fi
    RUN_DIR="${REPO_DIR}/exps/${EXP_NAME}"
fi

if [[ -z "$EXP_NAME" ]]; then
    EXP_NAME="$(basename "$RUN_DIR")"
fi

CKPT_DIR="${RUN_DIR}/checkpoints"
if [[ ! -d "$CKPT_DIR" ]]; then
    echo "Checkpoint directory not found: $CKPT_DIR" >&2
    exit 1
fi

if [[ -z "$STEPS" ]]; then
    STEP_LIST=()
    while IFS= read -r STEP_VALUE; do
        STEP_LIST+=("$STEP_VALUE")
    done < <(
        find "$CKPT_DIR" -maxdepth 1 -type f -name '*.pt' -print \
            | sed -E 's#^.*/([0-9]+)\.pt#\1#' \
            | awk -v max_step="$MAX_STEP" '$1 + 0 <= max_step' \
            | sort -n
    )
else
    IFS=',' read -r -a STEP_LIST <<< "$STEPS"
fi

if [[ ${#STEP_LIST[@]} -eq 0 ]]; then
    echo "No checkpoints selected under $CKPT_DIR" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU"

echo "================================================"
echo "Early checkpoint evaluation"
echo "run: $EXP_NAME"
echo "dir: $RUN_DIR"
echo "gpu/device: $GPU / $DEVICE"
echo "split: $SPLIT"
echo "steps: ${STEP_LIST[*]}"
echo "================================================"

for STEP in "${STEP_LIST[@]}"; do
    STEP="$(echo "$STEP" | xargs)"
    CKPT="${CKPT_DIR}/${STEP}.pt"
    if [[ ! -f "$CKPT" ]]; then
        echo "[skip] missing checkpoint: $CKPT"
        continue
    fi

    echo "------------------------------------------------"
    echo "checkpoint $STEP"
    echo "------------------------------------------------"

    if [[ "$SKIP_Q_PROBE" != "true" ]]; then
        Q_PROBE_CMD=(
            python "${SCRIPT_DIR}/q_scaffold_probe.py"
            --checkpoint "$CKPT" \
            --data-dir "$DATA_DIR" \
            --device "$DEVICE" \
            --split "$SPLIT" \
            --num-samples "$NUM_SAMPLES" \
            --batch-size "$BATCH_SIZE" \
            --num-workers "$NUM_WORKERS" \
            --timesteps "$TIMESTEPS" \
            --inference-dtype "$INFERENCE_DTYPE" \
            --patch-shuffle-mode "$PATCH_SHUFFLE_MODE"
        )
        if [[ -n "$LAYER_DEPTHS" ]]; then
            Q_PROBE_CMD+=(--layer-depths "$LAYER_DEPTHS")
        fi
        "${Q_PROBE_CMD[@]}"
    fi

    if [[ "$SKIP_SPATIAL" != "true" ]]; then
        SPATIAL_CMD=(
            python "${SCRIPT_DIR}/compute_spatial_metrics.py"
            --checkpoint "$CKPT" \
            --data-dir "$DATA_DIR" \
            --device "$DEVICE" \
            --split "$SPLIT" \
            --num-samples "$NUM_SAMPLES" \
            --batch-size "$BATCH_SIZE" \
            --num-workers "$NUM_WORKERS" \
            --timesteps "$TIMESTEPS"
        )
        if [[ -n "$LAYER_DEPTHS" ]]; then
            SPATIAL_CMD+=(--layer-depths "$LAYER_DEPTHS")
        fi
        "${SPATIAL_CMD[@]}"
    fi

    if [[ "$RUN_TEACHER_ALIGN" == "true" ]]; then
        python "${SCRIPT_DIR}/teacher_spatial_alignment.py" \
            --checkpoint "$CKPT" \
            --data-dir "$DATA_DIR" \
            --device "$DEVICE" \
            --split "$SPLIT" \
            --num-samples "$NUM_SAMPLES" \
            --batch-size "$BATCH_SIZE" \
            --num-workers "$NUM_WORKERS" \
            --timesteps "$TIMESTEPS" \
            --layer-depths "$TEACHER_LAYER_DEPTHS" \
            --inference-dtype "$INFERENCE_DTYPE" \
            --patch-shuffle-mode "$TEACHER_PATCH_SHUFFLE_MODE"
    fi

    if [[ "$RUN_LINEAR_PROBE" == "true" ]]; then
        LINEAR_LAYER="${LAYER_DEPTHS%%,*}"
        if [[ -z "$LINEAR_LAYER" ]]; then
            LINEAR_LAYER="4"
        fi
        python "${SCRIPT_DIR}/linear_probe_hidden.py" \
            --checkpoint "$CKPT" \
            --data-dir "$DATA_DIR" \
            --device "$DEVICE" \
            --batch-size "$BATCH_SIZE" \
            --num-workers "$NUM_WORKERS" \
            --timestep 0.0 \
            --layer-depth "$LINEAR_LAYER" \
            --max-train-samples "$NUM_SAMPLES" \
            --max-val-samples "$NUM_SAMPLES"
    fi
done

python "${SCRIPT_DIR}/collect_early_eval.py" --run-dir "$RUN_DIR"

if [[ "$RUN_FID" == "true" ]]; then
    if [[ -z "$CONFIG" ]]; then
        echo "FID requires --config so eval_ckpts.sh can infer model/resolution" >&2
        exit 1
    fi
    if [[ -z "$FID_STEPS" ]]; then
        LAST_INDEX=$((${#STEP_LIST[@]} - 1))
        FID_STEPS="${STEP_LIST[$LAST_INDEX]}"
    fi
    NUM_FID_SAMPLES="$NUM_FID_SAMPLES" \
    EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" \
    EVAL_NUM_STEPS="$EVAL_NUM_STEPS" \
    CFG_SCALE="$CFG_SCALE" \
    GUIDANCE_LOW="$GUIDANCE_LOW" \
    GUIDANCE_HIGH="$GUIDANCE_HIGH" \
    VAE="$VAE" \
    REF_BATCH="$REF_BATCH" \
    "${REPO_DIR}/eval_ckpts.sh" \
        --config "$CONFIG" \
        --exp-name "$EXP_NAME" \
        --gpu "$GPU" \
        --num-gpus "$(awk -F',' '{print NF}' <<< "$GPU")" \
        --steps "$FID_STEPS" \
        --cfg-scale "$CFG_SCALE" \
        --guidance-low "$GUIDANCE_LOW" \
        --guidance-high "$GUIDANCE_HIGH" \
        --vae "$VAE" \
        --ref-batch "$REF_BATCH"
fi

echo "Done. CSV summaries are under ${RUN_DIR}/early_eval."
