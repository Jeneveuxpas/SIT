#!/bin/bash
# -*- coding: utf-8 -*-
# 批量搜索 cfg-scale
# 用法: ./sweep_cfg.sh --guidance-high 0.65
#       ./sweep_cfg.sh --guidance-high 0.7
#       ./sweep_cfg.sh --guidance-high 0.75

CONFIG="configs/conv_3_kv_2.0.yaml"
EXP_NAME="conv_3_kv_2.0"
STEPS="0400000"
GUIDANCE_LOW="0.0"
GUIDANCE_HIGH=""
GPU="0,1,2,3,4,5,6,7"
NUM_GPUS="8" 

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --guidance-high) GUIDANCE_HIGH="$2"; shift 2 ;;
        --guidance-low)  GUIDANCE_LOW="$2"; shift 2 ;;
        --steps)         STEPS="$2"; shift 2 ;;
        --config)        CONFIG="$2"; shift 2 ;;
        --exp-name)      EXP_NAME="$2"; shift 2 ;;
        --gpu)           GPU="$2"; shift 2 ;;
        --num-gpus)      NUM_GPUS="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

if [ -z "$GUIDANCE_HIGH" ]; then
    echo "请指定 --guidance-high，例如: ./sweep_cfg.sh --guidance-high 0.65"
    exit 1
fi

# 要搜索的 cfg-scale 列表，按需修改
CFG_SCALES=(2.1)

echo "================================================"
echo "CFG-Scale 批量搜索"
echo "Config: ${CONFIG}"
echo "Exp: ${EXP_NAME}"
echo "Steps: ${STEPS}"
echo "Guidance: [${GUIDANCE_LOW}, ${GUIDANCE_HIGH}]"
echo "CFG scales: ${CFG_SCALES[*]}"
echo "总计: ${#CFG_SCALES[@]} 组"
echo "================================================"

for CFG in "${CFG_SCALES[@]}"; do
    echo ""
    echo "============ cfg-scale=${CFG} ============"
    ./eval_ckpts.sh \
        --config ${CONFIG} \
        --exp-name ${EXP_NAME} \
        --steps "${STEPS}" \
        --guidance-low ${GUIDANCE_LOW} \
        --guidance-high ${GUIDANCE_HIGH} \
        --cfg-scale ${CFG} \
        --gpu ${GPU} \
        --num-gpus ${NUM_GPUS}
    echo "============ cfg-scale=${CFG} 完成 ============"
    echo ""
done

echo "全部 cfg-scale 搜索完成！"
