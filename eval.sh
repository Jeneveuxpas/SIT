#!/bin/bash

# 限制只使用 GPU 1
export CUDA_VISIBLE_DEVICES=0

accelerate launch train.py --config configs/irepa.yaml \
  --model="SiT-B/2" \
  --enc-type="dinov2-vit-b" \
  --encoder-depth=6 \
  --data-dir=/home/jiacheng/imagenet_repa \
  --exp-name="sitb2-irepa-mse0"

sleep 5m

random_number=$((RANDOM % 100 + 1200))
NUM_GPUS=1
STEP="0100000"  # 修改为你的 checkpoint step
SAVE_PATH="/home/jiacheng/code/iREPA/ldm/exps/sitb2-irepa-mse0"
NUM_STEP=250
MODEL_SIZE='B'
CFG_SCALE=1.0
GH=1.0

export NCCL_P2P_DISABLE=1

torchrun --nnodes=1 --nproc_per_node=1 generate.py \
  --model SiT-B/2 \
  --num-fid-samples 50000 \
  --ckpt ${SAVE_PATH}/checkpoints/${STEP}.pt \
  --path-type=linear \
  --encoder-depth=6 \
  --per-proc-batch-size=64 \
  --mode=sde \
  --num-steps=${NUM_STEP} \
  --cfg-scale=${CFG_SCALE} \
  --guidance-high=${GH} \
  --sample-dir ${SAVE_PATH}/checkpoints \

# 计算 FID (需要准备 reference batch)
python ./evaluations/evaluator.py \
    --ref_batch /home/jiacheng/datasets/VIRTUAL_imagenet256_labeled.npz \
    --sample_batch /home/jiacheng/code/iREPA/ldm/exps/sitb2-irepa-mse0/checkpoints/sitb2-irepa-mse0_cfg1.0-seed0-modesde-steps250_0100000.npz \
    --save_path ${SAVE_PATH}/checkpoints \
    --cfg_cond 1 \
    --step ${STEP} \
    --num_steps ${NUM_STEP} \
    --cfg ${CFG_SCALE} \
    --gh ${GH}
