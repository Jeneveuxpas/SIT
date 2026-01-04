#!/bin/bash

# ============================================================================
# DINO-KV Distillation for iREPA 训练 + 自动评估脚本
# 
# 用法:
#   ./train_dinokv.sh [选项]
#
# 选项:
#   --dino-layers       DINO层索引 (默认: 8)
#   --sit-layers        SiT层索引 (默认: 8)
#   --stage1-ratio      Stage 1比例 (默认: 0.5)
#   --align-mode        对齐模式 (默认: logits_attn)可选: logits, logits_attn, attn_mse, kv_mse, k_only
#   --proj-coeff        REPA投影系数 (默认: 1.0)
#   --distill-coeff     蒸馏loss系数 (默认: 1.0)
#   --encoder-depth     REPA对齐层 (默认: 8)
#   --use-adaln-dino    启用DINO AdaLN调制 (可选)
#   --adaln-dropout     AdaLN dropout比例 (默认: 0.5)
#   --gpu               使用哪张GPU (默认: 0,1)
#
# 示例:
#   ./train_dinokv.sh --gpu 2,3 --dino-layers 9 --sit-layers 3 --encoder-depth 6 --align-mode attn_mse --proj-coeff 1.0 --distill-coeff 2.0 --max-steps 100000 --stage1-ratio 0.3
# ============================================================================

set -e

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --dino-layers)
            DINO_LAYERS_ARG="$2"
            shift 2
            ;;
        --sit-layers)
            SIT_LAYERS_ARG="$2"
            shift 2
            ;;
        --stage1-ratio)
            STAGE1_RATIO_ARG="$2"
            shift 2
            ;;
        --align-mode)
            ALIGN_MODE_ARG="$2"
            shift 2
            ;;
        --proj-coeff)
            PROJ_COEFF_ARG="$2"
            shift 2
            ;;
        --distill-coeff)
            DISTILL_COEFF_ARG="$2"
            shift 2
            ;;
        --encoder-depth)
            ENCODER_DEPTH_ARG="$2"
            shift 2
            ;;
        --resume-step)
            RESUME_STEP_ARG="$2"
            shift 2
            ;;
        --max-steps)
            MAX_STEPS_ARG="$2"
            shift 2
            ;;
        --gpu)
            GPU_ARG="$2"
            shift 2
            ;;
        --exp-name)
            EXP_NAME_ARG="$2"
            shift 2
            ;;
        --use-adaln-dino)
            USE_ADALN_DINO=true
            shift 1
            ;;
        --adaln-dropout)
            ADALN_DROPOUT_ARG="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

export CUDA_VISIBLE_DEVICES="${GPU_ARG:-0,1}"

# RTX 4000系列不支持P2P和IB通信，需要禁用
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# ============================================================================
# 模型配置
# ============================================================================
MODEL="SiT-B/2-DINOKV"
MODEL_SIZE="B"
ENCODER_TYPE="dinov2-vit-b"
ENCODER_DEPTH="${ENCODER_DEPTH_ARG:-8}"
Z_DIMS="768"

# ============================================================================
# DINO-KV 配置
# ============================================================================
DINO_LAYER_INDICES="${DINO_LAYERS_ARG:-8}"
SIT_LAYER_INDICES="${SIT_LAYERS_ARG:-8}"
STAGE1_RATIO="${STAGE1_RATIO_ARG:-0.3}"
ALIGN_MODE="${ALIGN_MODE_ARG:-logits_attn}"
PROJ_COEFF="${PROJ_COEFF_ARG:-1.0}"
DISTILL_COEFF="${DISTILL_COEFF_ARG:-1.0}"
PROJECTION_LAYER_TYPE="conv"
ADALN_DROPOUT="${ADALN_DROPOUT_ARG:-0.5}"

# ============================================================================
# 训练配置
# ============================================================================
BATCH_SIZE=256
GRADIENT_ACCUMULATION_STEPS=1
LEARNING_RATE=1e-4
MAX_STEPS="${MAX_STEPS_ARG:-400000}"
CHECKPOINT_STEPS=10000
SAMPLING_STEPS=200000
RESUME_STEP="${RESUME_STEP_ARG:-0}"

# ============================================================================
# 评估配置
# ============================================================================
EVAL_STEP=""  # 留空则评估最后一个checkpoint

# 采样配置
NUM_STEP=250
CFG_SCALE=1.0
GH=1.0
MODE="sde"  # sde or ode

# FID样本数量
NUM_FID_SAMPLES=50000

# 数据路径
DATA_DIR="/dev/shm/imagenet_repa"

# 参考数据集路径 (用于FID计算)
REF_BATCH="/disks/sata5/fuhan/VIRTUAL_imagenet256_labeled.npz"

# ============================================================================
# 自动生成实验名称
# ============================================================================
DINO_LAYERS_NAME=$(echo ${DINO_LAYER_INDICES} | tr ',' '_')
SIT_LAYERS_NAME=$(echo ${SIT_LAYER_INDICES} | tr ',' '_')

if [ -z "$EXP_NAME_ARG" ]; then
    EXP_NAME="dinokv_${MODEL_SIZE,,}_d${DINO_LAYERS_NAME}_s${SIT_LAYERS_NAME}_${ALIGN_MODE}_s1r${STAGE1_RATIO}"
    if [ "$PROJ_COEFF" != "1.0" ]; then
        EXP_NAME="${EXP_NAME}_repa${PROJ_COEFF}"
    fi
    if [ "$DISTILL_COEFF" != "1.0" ]; then
        EXP_NAME="${EXP_NAME}_dc${DISTILL_COEFF}"
    fi
else
    EXP_NAME="$EXP_NAME_ARG"
fi

echo "================================================"
echo "DINO-KV Distillation for iREPA"
echo "================================================"
echo "实验名称: ${EXP_NAME}"
echo "DINO层: ${DINO_LAYER_INDICES}"
echo "SiT层: ${SIT_LAYER_INDICES}"
echo "Stage1比例: ${STAGE1_RATIO}"
echo "对齐模式: ${ALIGN_MODE}"
echo "REPA系数: ${PROJ_COEFF}"
echo "蒸馏系数: ${DISTILL_COEFF}"
echo "AdaLN DINO: ${USE_ADALN_DINO:-false}"
echo "AdaLN Dropout: ${ADALN_DROPOUT}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "================================================"

# ============================================================================
# 开始训练
# ============================================================================

SAVE_PATH="exps/${EXP_NAME}"

# 计算 GPU 数量
GPU_COUNT=$(echo ${CUDA_VISIBLE_DEVICES} | tr ',' '\n' | wc -l)
echo "GPU 数量: ${GPU_COUNT}"

# 随机端口避免冲突
MASTER_PORT=$((RANDOM % 1000 + 29500))

if [ "$GPU_COUNT" -gt 1 ]; then
    MULTI_GPU_ARGS="--multi_gpu"
else
    MULTI_GPU_ARGS=""
fi
# 构建 AdaLN 参数
ADALN_ARGS=""
if [ "$USE_ADALN_DINO" = true ]; then
    ADALN_ARGS="--use-adaln-dino --adaln-dropout ${ADALN_DROPOUT}"
fi

# accelerate launch \
#     ${MULTI_GPU_ARGS} \
#     --main_process_port ${MASTER_PORT} \
#     --num_processes ${GPU_COUNT} \
#     --mixed_precision fp16 \
#     train_dinokv.py \
#     --exp-name ${EXP_NAME} \
#     --output-dir exps \
#     --data-dir ${DATA_DIR} \
#     --model ${MODEL} \
#     --enc-type ${ENCODER_TYPE} \
#     --encoder-depth ${ENCODER_DEPTH} \
#     --projection-layer-type ${PROJECTION_LAYER_TYPE} \
#     --dino-layer-indices ${DINO_LAYER_INDICES} \
#     --sit-layer-indices ${SIT_LAYER_INDICES} \
#     --stage1-ratio ${STAGE1_RATIO} \
#     --align-mode ${ALIGN_MODE} \
#     --proj-coeff ${PROJ_COEFF} \
#     --distill-coeff ${DISTILL_COEFF} \
#     --batch-size ${BATCH_SIZE} \
#     --gradient-accumulation-steps ${GRADIENT_ACCUMULATION_STEPS} \
#     --learning-rate ${LEARNING_RATE} \
#     --max-train-steps ${MAX_STEPS} \
#     --checkpointing-steps ${CHECKPOINT_STEPS} \
#     --sampling-steps ${SAMPLING_STEPS} \
#     --resume-step ${RESUME_STEP} \
#     --mixed-precision fp16 \
#     --allow-tf32 \
#     --repa-loss \
#     --spnorm-method zscore \
#     --num-workers 12 \
#     ${ADALN_ARGS}

# 检查训练是否成功

echo "================================================"
echo "训练完成！开始评估..."
echo "================================================"

# ============================================================================
# 自动评估
# ============================================================================

# 如果没有指定EVAL_STEP，找最新的checkpoint
if [ -z "$EVAL_STEP" ]; then
    LATEST_CKPT=$(ls ${SAVE_PATH}/checkpoints/*.pt 2>/dev/null | sort -t'/' -k3 -V | tail -1)
    if [ -z "$LATEST_CKPT" ]; then
        echo "未找到checkpoint，跳过评估"
        exit 0
    fi
    EVAL_STEP=$(basename ${LATEST_CKPT} .pt)
    echo "使用最新checkpoint: ${EVAL_STEP}"
fi

# 随机端口
random_number=$((RANDOM % 100 + 29600))

echo "================================================"
echo "生成 ${NUM_FID_SAMPLES} 个样本用于FID计算..."
echo "Checkpoint: ${SAVE_PATH}/checkpoints/${EVAL_STEP}.pt"
echo "================================================"

torchrun --nproc_per_node=${GPU_COUNT} --master_port=$random_number generate_dinokv.py \
    --model ${MODEL} \
    --num-fid-samples ${NUM_FID_SAMPLES} \
    --ckpt ${SAVE_PATH}/checkpoints/${EVAL_STEP}.pt \
    --path-type=linear \
    --encoder-depth=${ENCODER_DEPTH} \
    --z-dims=${Z_DIMS} \
    --dino-layer-indices=${DINO_LAYER_INDICES} \
    --sit-layer-indices=${SIT_LAYER_INDICES} \
    --per-proc-batch-size=128 \
    --mode=${MODE} \
    --num-steps=${NUM_STEP} \
    --cfg-scale=${CFG_SCALE} \
    --guidance-low=0.0 \
    --guidance-high=${GH} \
    --sample-dir ${SAVE_PATH}/checkpoints

echo "================================================"
echo "样本生成完成，计算FID等指标..."
echo "================================================"

# 检查 evaluator 是否存在
if [ ! -f "./evaluations/evaluator.py" ]; then
    echo "警告: ./evaluations/evaluator.py 不存在"
    echo "样本已生成到: ${SAVE_PATH}/checkpoints/"
    echo "请手动运行评估脚本计算FID"
    exit 0
fi

# 构建样本文件名 (格式: {exp_name}_{hparams})
# 注意: 这个格式需要与 generate_dinokv.py 的输出格式一致
SAMPLE_NPZ="${SAVE_PATH}/checkpoints/${EXP_NAME}_cfg${CFG_SCALE}-seed0-mode${MODE:-sde}-steps${NUM_STEP}_${EVAL_STEP}.npz"

if [ -f "$SAMPLE_NPZ" ]; then
    python ./evaluations/evaluator.py \
        --ref_batch ${REF_BATCH} \
        --sample_batch ${SAMPLE_NPZ} \
        --save_path ${SAVE_PATH}/checkpoints \
        --cfg_cond 1 \
        --step ${EVAL_STEP} \
        --num_steps ${NUM_STEP} \
        --cfg ${CFG_SCALE} \
        --gh ${GH}
else
    echo "未找到样本文件: ${SAMPLE_NPZ}"
    echo "请检查生成的样本路径"
fi

echo "================================================"
echo "全部完成！"
echo "================================================"
