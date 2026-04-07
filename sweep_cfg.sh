#!/bin/bash
# -*- coding: utf-8 -*-
# ============================================================================
# 批量搜索 cfg-scale 和 guidance-high
#
# 用法:
#   ./sweep_cfg.sh
#   ./sweep_cfg.sh --steps "0600000,0700000,0800000"
#   ./sweep_cfg.sh --cfg-scales "1.2,1.4,1.6,1.8" --guidance-highs "1.0,0.8,0.72,0.65"
#   ./sweep_cfg.sh --config configs/SiT-XL-early-stop-300.yaml --exp-name SIT-XL-early-stop-300 \
#       --steps "0600000,0700000,0800000" --vae ema --gpu 4,5,6,7 --num-gpus 4
# ============================================================================
set -e

# 默认参数
CONFIG="configs/SIT-XL.yaml"
EXP_NAME="SIT-XL"
STEPS="2000000"

GPU="0,1,2,3,4,5,6,7"
NUM_GPUS="8"
SEED="0"

MODE="sde"
VAE="mse"
GUIDANCE_LOW="0.0"
NUM_STEPS="250"
REF_BATCH="/workspace/SIT/VIRTUAL_imagenet256_labeled.npz"

# 默认 sweep 网格
CFG_SCALES_STR="1.2,1.4,1.6,1.8"
GUIDANCE_HIGHS_STR="1.0,0.8,0.72,0.65"

DRY_RUN="false"

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --exp-name)
            EXP_NAME="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --vae)
            VAE="$2"
            shift 2
            ;;
        --guidance-low)
            GUIDANCE_LOW="$2"
            shift 2
            ;;
        --guidance-high)
            GUIDANCE_HIGHS_STR="$2"
            shift 2
            ;;
        --guidance-highs)
            GUIDANCE_HIGHS_STR="$2"
            shift 2
            ;;
        --cfg-scale)
            CFG_SCALES_STR="$2"
            shift 2
            ;;
        --cfg-scales)
            CFG_SCALES_STR="$2"
            shift 2
            ;;
        --num-steps)
            NUM_STEPS="$2"
            shift 2
            ;;
        --ref-batch)
            REF_BATCH="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

IFS=',' read -r -a CFG_SCALES <<< "${CFG_SCALES_STR}"
IFS=',' read -r -a GUIDANCE_HIGHS <<< "${GUIDANCE_HIGHS_STR}"

if [ ${#CFG_SCALES[@]} -eq 0 ]; then
    echo "cfg-scale 列表不能为空"
    exit 1
fi

if [ ${#GUIDANCE_HIGHS[@]} -eq 0 ]; then
    echo "guidance-high 列表不能为空"
    exit 1
fi

TOTAL_JOBS=$(( ${#CFG_SCALES[@]} * ${#GUIDANCE_HIGHS[@]} ))

echo "================================================"
echo "CFG / Guidance 批量搜索"
echo "Config: ${CONFIG}"
echo "Exp: ${EXP_NAME}"
echo "Steps: ${STEPS}"
echo "GPU: ${GPU} (${NUM_GPUS} GPUs)"
echo "Seed: ${SEED}"
echo "Mode: ${MODE}"
echo "VAE: ${VAE}"
echo "Guidance low: ${GUIDANCE_LOW}"
echo "Guidance highs: ${GUIDANCE_HIGHS[*]}"
echo "CFG scales: ${CFG_SCALES[*]}"
echo "Num steps: ${NUM_STEPS}"
echo "总计: ${TOTAL_JOBS} 组"
echo "================================================"

JOB_ID=0
for GH in "${GUIDANCE_HIGHS[@]}"; do
    GH=$(echo "${GH}" | xargs)
    for CFG in "${CFG_SCALES[@]}"; do
        CFG=$(echo "${CFG}" | xargs)
        JOB_ID=$((JOB_ID + 1))

        echo ""
        echo "============ [${JOB_ID}/${TOTAL_JOBS}] cfg=${CFG}, gh=${GH} ============"

        CMD=(
            ./eval_ckpts.sh
            --config "${CONFIG}"
            --exp-name "${EXP_NAME}"
            --steps "${STEPS}"
            --guidance-low "${GUIDANCE_LOW}"
            --guidance-high "${GH}"
            --cfg-scale "${CFG}"
            --num-steps "${NUM_STEPS}"
            --mode "${MODE}"
            --vae "${VAE}"
            --seed "${SEED}"
            --gpu "${GPU}"
            --num-gpus "${NUM_GPUS}"
            --ref-batch "${REF_BATCH}"
        )

        printf 'Command:'
        printf ' %q' "${CMD[@]}"
        printf '\n'

        if [ "${DRY_RUN}" = "false" ]; then
            "${CMD[@]}"
        fi

        echo "============ [${JOB_ID}/${TOTAL_JOBS}] cfg=${CFG}, gh=${GH} 完成 ============"
        echo ""
    done
done

echo "全部 sweep 完成！"
