#!/bin/bash
# -*- coding: utf-8 -*-
# ============================================================================
# 批量评估中间 checkpoints
# 
# 用法:
#   # 评估指定的 steps
#   ./eval_ckpts.sh --config configs/SIT-XL.yaml --exp-name SIT-XL --steps "0800000,0850000,0900000,0950000,1000000,1050000" --gpu 0,1,2,3 --num-gpus 4
#
#   # 使用 guidance interval
#   ./eval_ckpts.sh --config configs/SIT-XL.yaml --exp-name SIT-XL --steps "1000000" --guidance-low 0.0 --guidance-high 0.7 --cfg-scale 1.65
#   ./eval_ckpts.sh --config configs/SIT-XL.yaml --exp-name SIT-XL --steps "1000000" --guidance-low 0.0 --guidance-high 0.7 --cfg-scale 1.7
#   ./eval_ckpts.sh --config configs/SIT-XL.yaml --exp-name SIT-XL --steps "1000000" --guidance-low 0.0 --guidance-high 0.7 --cfg-scale 1.8
#   ./eval_ckpts.sh --config configs/SIT-XL.yaml --exp-name SIT-XL --steps "1000000" --guidance-low 0.0 --guidance-high 0.65 --cfg-scale 1.65
#   ./eval_ckpts.sh --config configs/SIT-XL.yaml --exp-name SIT-XL --steps "1000000" --guidance-low 0.0 --guidance-high 0.65 --cfg-scale 1.7
#   ./eval_ckpts.sh --config configs/SIT-XL.yaml --exp-name SIT-XL --steps "1000000" --guidance-low 0.0 --guidance-high 0.65 --cfg-scale 1.8
#   # 评估所有 checkpoints
#   ./eval_ckpts.sh --config configs/default.yaml --exp-name my_exp --all
#
#   # 评估某个范围内的 checkpoints
#   ./eval_ckpts.sh --config configs/default.yaml --exp-name my_exp --from 50000 --to 200000
#
#   # 评估最近 N 个 checkpoints
#   ./eval_ckpts.sh --config configs/default.yaml --exp-name my_exp --last 5
# ============================================================================
set -e

# 默认参数
GPU="${GPU:-0,1,2,3,4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-8}"

# FID 评估参数
NUM_FID_SAMPLES="${NUM_FID_SAMPLES:-50000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
EVAL_NUM_STEPS="${EVAL_NUM_STEPS:-250}"
CFG_SCALE="${CFG_SCALE:-1.0}"
GUIDANCE_LOW="${GUIDANCE_LOW:-0.0}"
GUIDANCE_HIGH="${GUIDANCE_HIGH:-1.0}"
MODE="${MODE:-sde}"
REF_BATCH="${REF_BATCH:-/workspace/SIT/VIRTUAL_imagenet256_labeled.npz}"

# 解析命令行参数
EVAL_ALL="false"
EVAL_LAST=""
FROM_STEP=""
TO_STEP=""
STEPS=""

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
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --all)
            EVAL_ALL="true"
            shift
            ;;
        --last)
            EVAL_LAST="$2"
            shift 2
            ;;
        --from)
            FROM_STEP="$2"
            shift 2
            ;;
        --to)
            TO_STEP="$2"
            shift 2
            ;;
        --cfg-scale)
            CFG_SCALE="$2"
            shift 2
            ;;
        --num-steps)
            EVAL_NUM_STEPS="$2"
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
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 检查必要参数
if [ -z "$EXP_NAME" ]; then
    echo "请指定实验名: --exp-name <name>"
    exit 1
fi

SAVE_PATH="exps/${EXP_NAME}"
CKPT_DIR="${SAVE_PATH}/checkpoints"
export CUDA_VISIBLE_DEVICES="${GPU}"

# 检查 checkpoint 目录
if [ ! -d "$CKPT_DIR" ]; then
    echo "Checkpoint 目录不存在: ${CKPT_DIR}"
    exit 1
fi

# ============================================================================
# 收集要评估的 checkpoints
# ============================================================================
CKPT_LIST=()

if [ "$EVAL_ALL" = "true" ]; then
    # 评估所有 checkpoints
    for ckpt in ${CKPT_DIR}/*.pt; do
        step=$(basename ${ckpt} .pt)
        # 过滤掉非数字的文件名
        if [[ $step =~ ^[0-9]+$ ]]; then
            CKPT_LIST+=($step)
        fi
    done
elif [ -n "$EVAL_LAST" ]; then
    # 评估最近 N 个
    for ckpt in $(ls ${CKPT_DIR}/*.pt 2>/dev/null | sort -t'/' -k3 -V | tail -${EVAL_LAST}); do
        step=$(basename ${ckpt} .pt)
        if [[ $step =~ ^[0-9]+$ ]]; then
            CKPT_LIST+=($step)
        fi
    done
elif [ -n "$FROM_STEP" ] || [ -n "$TO_STEP" ]; then
    # 评估某个范围内的
    FROM_STEP=${FROM_STEP:-0}
    TO_STEP=${TO_STEP:-999999999}
    for ckpt in ${CKPT_DIR}/*.pt; do
        step=$(basename ${ckpt} .pt)
        if [[ $step =~ ^[0-9]+$ ]] && [ $step -ge $FROM_STEP ] && [ $step -le $TO_STEP ]; then
            CKPT_LIST+=($step)
        fi
    done
elif [ -n "$STEPS" ]; then
    # 使用指定的 steps
    IFS=',' read -ra CKPT_LIST <<< "$STEPS"
else
    echo "请指定要评估的 checkpoints:"
    echo "  --steps \"50000,100000\"  指定具体步数"
    echo "  --all                    评估所有"
    echo "  --last N                 评估最近 N 个"
    echo "  --from X --to Y          评估范围"
    exit 1
fi

# 排序
IFS=$'\n' CKPT_LIST=($(sort -n <<<"${CKPT_LIST[*]}")); unset IFS

echo "================================================"
echo "批量评估 Checkpoints"
echo "实验名: ${EXP_NAME}"
echo "GPU: ${GPU} (${NUM_GPUS} GPUs)"
echo "Guidance interval: [${GUIDANCE_LOW}, ${GUIDANCE_HIGH}]"
echo "待评估 checkpoints: ${CKPT_LIST[*]}"
echo "总计: ${#CKPT_LIST[@]} 个"
echo "================================================"

# ============================================================================
# 解析模型参数
# ============================================================================
if [ -n "$CONFIG" ]; then
    MODEL=$(grep "^model:" $CONFIG | awk '{print $2}')
    MODEL=${MODEL%-EncoderKV}
    echo "从配置中检测到模型: ${MODEL}"
else
    MODEL="SiT-B/2"
    echo "使用默认模型: ${MODEL}"
fi

# ============================================================================
# 评估每个 checkpoint
# ============================================================================
MASTER_PORT=$((29500 + RANDOM % 1000))
RESULTS_FILE="${SAVE_PATH}/eval_results.txt"

echo "" >> ${RESULTS_FILE}
echo "========== 评估开始 $(date) ==========" >> ${RESULTS_FILE}

for STEP in "${CKPT_LIST[@]}"; do
    STEP=$(echo $STEP | xargs)  # 去除空格
    CKPT_FILE="${CKPT_DIR}/${STEP}.pt"
    
    if [ ! -f "$CKPT_FILE" ]; then
        echo "[跳过] Checkpoint 不存在: ${CKPT_FILE}"
        continue
    fi

    echo "------------------------------------------------"
    echo "[${STEP}] 开始评估..."
    echo "------------------------------------------------"

    # 生成样本
    torchrun --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} generate.py \
        --ckpt ${CKPT_FILE} \
        --num-fid-samples ${NUM_FID_SAMPLES} \
        --per-proc-batch-size ${EVAL_BATCH_SIZE} \
        --mode ${MODE} \
        --model ${MODEL} \
        --num-steps ${EVAL_NUM_STEPS} \
        --cfg-scale ${CFG_SCALE} \
        --guidance-low ${GUIDANCE_LOW} \
        --guidance-high ${GUIDANCE_HIGH} \
        --sample-dir ${CKPT_DIR}

    # 构建样本文件名（与 generate.py 中的 cfg_intv 逻辑一致）
    if [ "${GUIDANCE_LOW}" = "0.0" ] && [ "${GUIDANCE_HIGH}" = "1.0" ]; then
        CFG_INTV=""
    else
        CFG_INTV="_${GUIDANCE_LOW}_${GUIDANCE_HIGH}"
    fi
    SAMPLE_NPZ="${CKPT_DIR}/${EXP_NAME}_cfg${CFG_SCALE}${CFG_INTV}-seed0-mode${MODE}-steps${EVAL_NUM_STEPS}_${STEP}.npz"

    if [ -f "$SAMPLE_NPZ" ]; then
        echo "计算 FID..."
        python evaluations/evaluator.py \
            --ref_batch ${REF_BATCH} \
            --sample_batch ${SAMPLE_NPZ} \
            --save_path ${CKPT_DIR} \
            --step ${STEP} \
            --num_steps ${EVAL_NUM_STEPS} \
            --cfg ${CFG_SCALE}
        
        echo "[${STEP}] 评估完成" | tee -a ${RESULTS_FILE}
    else
        echo "[${STEP}] 未找到样本文件: ${SAMPLE_NPZ}" | tee -a ${RESULTS_FILE}
    fi
done

echo "================================================"
echo "全部评估完成！"
echo "结果保存在: ${SAVE_PATH}"
echo "================================================"
