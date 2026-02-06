#!/bin/bash
# -*- coding: utf-8 -*-
# ============================================================================
# iREPA 统一启动脚本 - 训练 + 评估
# 
# 用法: 
#   ./launch.sh --config configs/default.yaml --exp-name my_exp
#   ./launch.sh --config configs/kv_coeff-5.0.yaml --exp-name kv_coeff-5.0 --gpu 6,7 --num-gpus 2
# ============================================================================
set -e

# 默认参数
GPU="${GPU:-0,1,2,3,4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-8}"
RESUME_STEP="${RESUME_STEP:-0}"  # 恢复训练的步数，0表示从头开始
EVAL_STEPS="${EVAL_STEPS:-}"  # 评估的 checkpoint steps，逗号分隔，留空则评估最新
EVAL_ONLY="${EVAL_ONLY:-false}"
SKIP_EVAL="${SKIP_EVAL:-false}"

# FID 评估参数
NUM_FID_SAMPLES="${NUM_FID_SAMPLES:-50000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
EVAL_NUM_STEPS="${EVAL_NUM_STEPS:-250}"
CFG_SCALE="${CFG_SCALE:-1.0}"
MODE="${MODE:-sde}"
REF_BATCH="${REF_BATCH:-/workspace/SIT/VIRTUAL_imagenet256_labeled.npz}"

# 解析命令行参数
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
        --eval-only)
            EVAL_ONLY="true"
            shift
            ;;
        --skip-eval)
            SKIP_EVAL="true"
            shift
            ;;
        --eval-steps)
            EVAL_STEPS="$2"
            shift 2
            ;;
        --cfg-scale)
            CFG_SCALE="$2"
            shift 2
            ;;
        --resume-step)
            RESUME_STEP="$2"
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
export CUDA_VISIBLE_DEVICES="${GPU}"

# +
# ============================================================================
# 训练阶段
# ============================================================================
if [ "$EVAL_ONLY" = "false" ]; then
    echo "================================================"
    echo "开始训练..."
    echo "实验名: ${EXP_NAME}"
    echo "GPU: ${GPU} (${NUM_GPUS} GPUs)"
    if [ -n "$CONFIG" ]; then
        echo "配置文件: ${CONFIG}"
    fi
    echo "================================================"

    # 构建训练命令
    MASTER_PORT=$((29500 + RANDOM % 1000))
    TRAIN_CMD="accelerate launch --main_process_port ${MASTER_PORT} --num_processes ${NUM_GPUS} train.py --exp-name ${EXP_NAME}"

    if [ -n "$CONFIG" ]; then
        TRAIN_CMD="${TRAIN_CMD} --config ${CONFIG}"
    fi

    # 添加 resume-step 参数（如果不是0）
    if [ "$RESUME_STEP" -gt 0 ]; then
        TRAIN_CMD="${TRAIN_CMD} --resume-step ${RESUME_STEP}"
        echo "从 step ${RESUME_STEP} 恢复训练"
    fi

    # 执行训练
   eval ${TRAIN_CMD}

    echo "================================================"
    echo "训练完成！"
    echo "================================================"
fi

# ============================================================================
# 解析模型参数 (用于评估)
# ============================================================================
if [ -n "$CONFIG" ]; then
    # 从 config 中提取 model 字段
    # 假设格式: model: SiT-XL/2-EncoderKV
    MODEL=$(grep "^model:" $CONFIG | awk '{print $2}')
    
    # 去除 -EncoderKV 后缀 (如果存在)
    MODEL=${MODEL%-EncoderKV}
    
    echo "从配置中检测到模型: ${MODEL}"
else
    # 默认模型
    MODEL="SiT-B/2"
    echo "使用默认模型: ${MODEL}"
fi
# -

# ============================================================================
# 评估阶段
# ============================================================================
if [ "$SKIP_EVAL" = "false" ]; then
    echo "================================================"
    echo "开始评估..."
    echo "================================================"

    # 确定要评估的 checkpoints
    if [ -z "$EVAL_STEPS" ]; then
        # 如果没有指定，找最新的 checkpoint
        LATEST_CKPT=$(ls ${SAVE_PATH}/checkpoints/*.pt 2>/dev/null | sort -t'/' -k3 -V | tail -1)
        if [ -z "$LATEST_CKPT" ]; then
            echo "未找到 checkpoint"
            exit 1
        fi
        EVAL_STEPS=$(basename ${LATEST_CKPT} .pt)
        echo "使用最新 checkpoint: ${EVAL_STEPS}"
    fi

    # 遍历每个 checkpoint 进行评估
    IFS=',' read -ra STEPS_ARRAY <<< "$EVAL_STEPS"
    for STEP in "${STEPS_ARRAY[@]}"; do
        STEP=$(echo $STEP | xargs)  # 去除空格
        echo "------------------------------------------------"
        echo "评估 checkpoint: ${STEP}"
        echo "------------------------------------------------"

        # 生成样本
        torchrun --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} generate.py \
            --ckpt ${SAVE_PATH}/checkpoints/${STEP}.pt \
            --num-fid-samples ${NUM_FID_SAMPLES} \
            --per-proc-batch-size ${EVAL_BATCH_SIZE} \
            --mode ${MODE} \
            --model ${MODEL} \
            --num-steps ${EVAL_NUM_STEPS} \
            --cfg-scale ${CFG_SCALE} \
            --sample-dir ${SAVE_PATH}/checkpoints

        # 构建样本文件名
        SAMPLE_NPZ="${SAVE_PATH}/checkpoints/${EXP_NAME}_cfg${CFG_SCALE}-seed0-mode${MODE}-steps${EVAL_NUM_STEPS}_${STEP}.npz"

        if [ -f "$SAMPLE_NPZ" ]; then
            echo "计算 FID..."
            python evaluations/evaluator.py \
                --ref_batch ${REF_BATCH} \
                --sample_batch ${SAMPLE_NPZ} \
                --save_path ${SAVE_PATH}/checkpoints \
                --step ${STEP} \
                --num_steps ${EVAL_NUM_STEPS} \
                --cfg ${CFG_SCALE}
        else
            echo "未找到样本文件: ${SAMPLE_NPZ}"
        fi
    done
fi

echo "================================================"
echo "全部完成！"
echo "结果保存在: ${SAVE_PATH}"
echo "================================================"
