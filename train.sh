#!/bin/bash

# ============================================================================
# Encoder-KV Distillation for iREPA 训练 + 自动评估脚本
# 
# 用法:
#   ./train.sh [选项]
#
# 选项:
#   --enc-layers        Encoder层索引 (默认: 9, 1-based)
#   --sit-layers        SiT层索引 (默认: 5, 1-based)
#   --stage1-ratio      Stage 1比例 (默认: 0.5)
#   --align-mode        对齐模式 (默认: logits_attn)可选: logits, logits_attn, attn_mse, kv_mse, k_only, attn_kl
#   --proj-coeff        REPA投影系数 (默认: 1.0)
#   --distill-coeff     蒸馏loss系数 (默认: 1.0)
#   --encoder-depth     REPA对齐层 (默认: 8)
#   --model-size        模型大小 B/L/XL (默认: B)
#   --gpu               使用哪张GPU (默认: 0,1)
#
# 示例:
#./train.sh --gpu 0,1 --model-size XL --enc-layers 9 --sit-layers 5 --encoder-depth 10 --projection-loss-type mse_v --proj-coeff 1.0 --max-steps 100000 --stage1-ratio 0.3 --distill-coeff 2.0 --align-mode attn_mse --kv-proj-type conv --kv-norm-type zscore
# ============================================================================
set -e

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --enc-layers)
            ENC_LAYERS_ARG="$2"
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
        --projection-loss-type)
            PROJECTION_LOSS_TYPE_ARG="$2"
            shift 2
            ;;
        --kv-proj-type)
            KV_PROJ_TYPE_ARG="$2"
            shift 2
            ;;
        --kv-norm-type)
            KV_NORM_TYPE_ARG="$2"
            shift 2
            ;;
        --kv-zscore-alpha)
            KV_ZSCORE_ALPHA_ARG="$2"
            shift 2
            ;;
        --no-repa-loss)
            NO_REPA_LOSS=true
            shift
            ;;
        --projection-layer-type)
            PROJECTION_LAYER_TYPE_ARG="$2"
            shift 2
            ;;
        --distill-t-threshold)
            DISTILL_T_THRESHOLD_ARG="$2"
            shift 2
            ;;
        --kv-mode)
            KV_MODE_ARG="$2"
            shift 2
            ;;
        --model-size)
            MODEL_SIZE_ARG="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

export CUDA_VISIBLE_DEVICES="${GPU_ARG:-0,1}"

# ============================================================================
# 模型配置
# ============================================================================
MODEL_SIZE="${MODEL_SIZE_ARG:-B}"
# 根据 MODEL_SIZE 设置模型和隐藏层维度
case "$MODEL_SIZE" in
    B|b)
        MODEL_SIZE="B"
        MODEL="SiT-B/2-EncoderKV"
        HIDDEN_SIZE=768
        ;;
    L|l)
        MODEL_SIZE="L"
        MODEL="SiT-L/2-EncoderKV"
        HIDDEN_SIZE=1024
        ;;
    XL|xl)
        MODEL_SIZE="XL"
        MODEL="SiT-XL/2-EncoderKV"
        HIDDEN_SIZE=1152
        ;;
    *)
        echo "无效的模型大小: $MODEL_SIZE, 请使用 B, L, 或 XL"
        exit 1
        ;;
esac
# DINO encoder 继续使用 dinov2-vit-b
ENCODER_TYPE="dinov2-vit-b"
ENCODER_DEPTH="${ENCODER_DEPTH_ARG:-8}"
Z_DIMS="768"

# ============================================================================
# Encoder-KV 配置
# ============================================================================
ENC_LAYER_INDICES="${ENC_LAYERS_ARG:-8}"
SIT_LAYER_INDICES="${SIT_LAYERS_ARG:-8}"
STAGE1_RATIO="${STAGE1_RATIO_ARG:-0.3}"
ALIGN_MODE="${ALIGN_MODE_ARG:-logits_attn}"
PROJ_COEFF="${PROJ_COEFF_ARG:-1.0}"
DISTILL_COEFF="${DISTILL_COEFF_ARG:-1.0}"
DISTILL_T_THRESHOLD="${DISTILL_T_THRESHOLD_ARG:-1.0}"
KV_MODE="${KV_MODE_ARG:-kv}"
PROJECTION_LOSS_TYPE="${PROJECTION_LOSS_TYPE_ARG:-cosine}"
KV_NORM_TYPE="${KV_NORM_TYPE_ARG:-layernorm}"
KV_ZSCORE_ALPHA="${KV_ZSCORE_ALPHA_ARG:-1.0}"
KV_PROJ_TYPE="${KV_PROJ_TYPE_ARG:-conv}"
PROJECTION_LAYER_TYPE="${PROJECTION_LAYER_TYPE_ARG:-conv}"

# 设置 REPA loss 参数
if [ "$NO_REPA_LOSS" = "true" ]; then
    REPA_LOSS_ARG="--no-repa-loss"
else
    REPA_LOSS_ARG="--repa-loss"
fi

# ============================================================================
# 训练配置
# ============================================================================
BATCH_SIZE=256
GRADIENT_ACCUMULATION_STEPS=1
LEARNING_RATE=1e-4
MAX_STEPS="${MAX_STEPS_ARG:-400000}"
CHECKPOINT_STEPS=10000
SAMPLING_STEPS=10000
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
ENC_LAYERS_NAME=$(echo ${ENC_LAYER_INDICES} | tr ',' '_')
SIT_LAYERS_NAME=$(echo ${SIT_LAYER_INDICES} | tr ',' '_')

if [ -z "$EXP_NAME_ARG" ]; then
    EXP_NAME="encoderkv_${MODEL_SIZE,,}_e${ENC_LAYERS_NAME}_s${SIT_LAYERS_NAME}_r${ENCODER_DEPTH}_${ALIGN_MODE}_s1r${STAGE1_RATIO}"
    # Add projection loss type (skip default cosine)
    if [ "$PROJECTION_LOSS_TYPE" != "mse_v" ]; then
        EXP_NAME="${EXP_NAME}_loss${PROJECTION_LOSS_TYPE}"
    fi
    if [ "$KV_PROJ_TYPE" != "conv" ]; then
        EXP_NAME="${EXP_NAME}_proj${KV_PROJ_TYPE}"
    fi
    # Add kv norm type (skip default layernorm)
    if [ "$KV_NORM_TYPE" != "layernorm" ]; then
        EXP_NAME="${EXP_NAME}_kvnorm${KV_NORM_TYPE}"
    fi
    # Add kv zscore alpha to name (skip default 1.0)
    if [ "$KV_ZSCORE_ALPHA" != "1.0" ]; then
        EXP_NAME="${EXP_NAME}_alpha${KV_ZSCORE_ALPHA}"
    fi
    if [ "$PROJ_COEFF" != "1.0" ]; then
        EXP_NAME="${EXP_NAME}_repa${PROJ_COEFF}"
    fi
    if [ "$DISTILL_COEFF" != "2.0" ]; then
        EXP_NAME="${EXP_NAME}_dc${DISTILL_COEFF}"
    fi
    # Add distill t threshold to name (skip default 1.0)
    if [ "$DISTILL_T_THRESHOLD" != "1.0" ]; then
        EXP_NAME="${EXP_NAME}_dt${DISTILL_T_THRESHOLD}"
    fi
    # Add kv mode to name (skip default kv)
    if [ "$KV_MODE" != "kv" ]; then
        EXP_NAME="${EXP_NAME}_${KV_MODE}"
    fi
else
    EXP_NAME="$EXP_NAME_ARG"
fi

echo "================================================"
echo "Encoder-KV Distillation for iREPA"
echo "================================================"
echo "实验名称: ${EXP_NAME}"
echo "Encoder层: ${ENC_LAYER_INDICES}"
echo "SiT层: ${SIT_LAYER_INDICES}"
echo "Stage1比例: ${STAGE1_RATIO}"
echo "对齐模式: ${ALIGN_MODE}"
echo "模型: ${MODEL}"
echo "REPA系数: ${PROJ_COEFF}"
echo "蒸馏系数: ${DISTILL_COEFF}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "================================================"

# ============================================================================
# 开始训练
# ============================================================================

SAVE_PATH="exps/${EXP_NAME}"

# 计算 GPU 数量
GPU_COUNT=$(echo ${CUDA_VISIBLE_DEVICES} | tr ',' '\n' | wc -l)
echo "GPU 数量: ${GPU_COUNT}"

# 随机端口避免冲突 (使用更大范围 + PID 确保唯一性)
MASTER_PORT=$((RANDOM % 10000 + 20000 + $$))
# 确保端口在有效范围内
MASTER_PORT=$((MASTER_PORT % 10000 + 20000))

# 生成唯一的运行ID，避免多进程 torchrun rendezvous 冲突
export TORCHELASTIC_RUN_ID="${EXP_NAME}_train_$$_$(date +%s)"

if [ "$GPU_COUNT" -gt 1 ]; then
    MULTI_GPU_ARGS="--multi_gpu"
else
    MULTI_GPU_ARGS=""
fi

accelerate launch \
    ${MULTI_GPU_ARGS} \
    --main_process_port ${MASTER_PORT} \
    --num_processes ${GPU_COUNT} \
    --mixed_precision fp16 \
    train.py \
    --exp-name ${EXP_NAME} \
    --output-dir exps \
    --data-dir ${DATA_DIR} \
    --model ${MODEL} \
    --enc-type ${ENCODER_TYPE} \
    --encoder-depth ${ENCODER_DEPTH} \
    --projection-layer-type conv \
    --proj-coeff ${PROJ_COEFF} \
    --projection-loss-type ${PROJECTION_LOSS_TYPE} \
    --enc-layer-indices ${ENC_LAYER_INDICES} \
    --sit-layer-indices ${SIT_LAYER_INDICES} \
    --stage1-ratio ${STAGE1_RATIO} \
    --align-mode ${ALIGN_MODE} \
    --kv-norm-type ${KV_NORM_TYPE} \
    --kv-zscore-alpha ${KV_ZSCORE_ALPHA} \
    --kv-proj-type ${KV_PROJ_TYPE} \
    --distill-coeff ${DISTILL_COEFF} \
    --distill-t-threshold ${DISTILL_T_THRESHOLD} \
    --kv-mode ${KV_MODE} \
    --batch-size ${BATCH_SIZE} \
    --gradient-accumulation-steps ${GRADIENT_ACCUMULATION_STEPS} \
    --learning-rate ${LEARNING_RATE} \
    --max-train-steps ${MAX_STEPS} \
    --checkpointing-steps ${CHECKPOINT_STEPS} \
    --sampling-steps ${SAMPLING_STEPS} \
    --resume-step ${RESUME_STEP} \
    --mixed-precision fp16 \
    --allow-tf32 \
    ${REPA_LOSS_ARG} \
    --spnorm-method zscore_spatial \
    --num-workers 12

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

# 随机端口（使用与训练不同的范围，避免冲突）
random_number=$((RANDOM % 10000 + 40000 + $$))
random_number=$((random_number % 10000 + 40000))

# 生成唯一的推理运行ID
export TORCHELASTIC_RUN_ID="${EXP_NAME}_infer_$$_$(date +%s)"

echo "================================================"
echo "生成 ${NUM_FID_SAMPLES} 个样本用于FID计算..."
echo "Checkpoint: ${SAVE_PATH}/checkpoints/${EVAL_STEP}.pt"
echo "================================================"

# Switch to standard SiT generator for clean evaluation
GENERATION_MODEL=${MODEL%-EncoderKV}

torchrun --nproc_per_node=${GPU_COUNT} --master_port=$random_number generate.py \
    --model ${GENERATION_MODEL} \
    --num-fid-samples ${NUM_FID_SAMPLES} \
    --ckpt ${SAVE_PATH}/checkpoints/${EVAL_STEP}.pt \
    --path-type=linear \
    --encoder-depth=${ENCODER_DEPTH} \
    --z-dims=${Z_DIMS} \
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
