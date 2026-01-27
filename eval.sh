#!/bin/bash
# ============================================================================
# 评估脚本 - 用于从 checkpoint 生成样本并计算 FID
# 
# 用法: ./eval.sh --exp-name my_exp --step 100000
# ============================================================================
set -e

# 默认参数
GPU="${GPU:-0}"
NUM_FID_SAMPLES="${NUM_FID_SAMPLES:-50000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_STEPS="${NUM_STEPS:-250}"
CFG_SCALE="${CFG_SCALE:-1.0}"
MODE="${MODE:-sde}"
MODEL="${MODEL:-SiT-B/2}"
REF_BATCH="${REF_BATCH:-/disks/sata5/fuhan/VIRTUAL_imagenet256_labeled.npz}"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --exp-name)
            EXP_NAME="$2"
            shift 2
            ;;
        --step)
            STEP="$2"
            shift 2
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --cfg-scale)
            CFG_SCALE="$2"
            shift 2
            ;;
        --num-steps)
            NUM_STEPS="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
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

# 如果没有指定 step，找最新的 checkpoint
if [ -z "$STEP" ]; then
    LATEST_CKPT=$(ls ${SAVE_PATH}/checkpoints/*.pt 2>/dev/null | sort -t'/' -k3 -V | tail -1)
    if [ -z "$LATEST_CKPT" ]; then
        echo "未找到 checkpoint"
        exit 1
    fi
    STEP=$(basename ${LATEST_CKPT} .pt)
    echo "使用最新 checkpoint: ${STEP}"
fi

export CUDA_VISIBLE_DEVICES="${GPU}"

echo "================================================"
echo "生成 ${NUM_FID_SAMPLES} 个样本..."
echo "实验: ${EXP_NAME}"
echo "Checkpoint: ${STEP}"
echo "GPU: ${GPU}"
echo "================================================"

# 生成样本 (encoder-depth 不再需要，推理时不使用)
torchrun --nproc_per_node=1 generate.py \
    --model ${MODEL} \
    --ckpt ${SAVE_PATH}/checkpoints/${STEP}.pt \
    --num-fid-samples ${NUM_FID_SAMPLES} \
    --per-proc-batch-size ${BATCH_SIZE} \
    --mode ${MODE} \
    --num-steps ${NUM_STEPS} \
    --cfg-scale ${CFG_SCALE} \
    --sample-dir ${SAVE_PATH}/checkpoints

echo "================================================"
echo "计算 FID..."
echo "================================================"

# 构建样本文件名
SAMPLE_NPZ="${SAVE_PATH}/checkpoints/${EXP_NAME}_cfg${CFG_SCALE}-seed0-mode${MODE}-steps${NUM_STEPS}_${STEP}.npz"

if [ -f "$SAMPLE_NPZ" ]; then
    python evaluations/evaluator.py \
        --ref_batch ${REF_BATCH} \
        --sample_batch ${SAMPLE_NPZ} \
        --save_path ${SAVE_PATH}/checkpoints \
        --step ${STEP} \
        --num_steps ${NUM_STEPS} \
        --cfg ${CFG_SCALE}
else
    echo "未找到样本文件: ${SAMPLE_NPZ}"
    echo "尝试查找其他样本文件..."
    ls ${SAVE_PATH}/checkpoints/*.npz 2>/dev/null || echo "未找到任何 .npz 文件"
fi

echo "================================================"
echo "评估完成！"
echo "================================================"
