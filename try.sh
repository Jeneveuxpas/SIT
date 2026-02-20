#!/bin/bash
set -e
./eval_ckpts.sh --config configs/attn_mse_repa_early_stop_500.yaml --exp-name attn_mse_repa_early_stop_500 --steps "0740000" --guidance-low 0.0 --guidance-high 0.7 --cfg-scale 1.65
./eval_ckpts.sh --config configs/attn_mse_repa_early_stop_500.yaml --exp-name attn_mse_repa_early_stop_500 --steps "0740000" --guidance-low 0.0 --guidance-high 0.7 --cfg-scale 1.7
./eval_ckpts.sh --config configs/attn_mse_repa_early_stop_500.yaml --exp-name attn_mse_repa_early_stop_500 --steps "0740000" --guidance-low 0.0 --guidance-high 0.7 --cfg-scale 1.8
./eval_ckpts.sh --config configs/SIT-XL.yaml --exp-name SIT-XL --steps "1000000" --guidance-low 0.0 --guidance-high 0.7 --cfg-scale 1.65
./eval_ckpts.sh --config configs/SIT-XL.yaml --exp-name SIT-XL --steps "1000000" --guidance-low 0.0 --guidance-high 0.7 --cfg-scale 1.7
./eval_ckpts.sh --config configs/SIT-XL.yaml --exp-name SIT-XL --steps "1000000" --guidance-low 0.0 --guidance-high 0.7 --cfg-scale 1.8
./eval_ckpts.sh --config configs/SIT-XL.yaml --exp-name SIT-XL --steps "1000000" --guidance-low 0.0 --guidance-high 0.65 --cfg-scale 1.65
./eval_ckpts.sh --config configs/SIT-XL.yaml --exp-name SIT-XL --steps "1000000" --guidance-low 0.0 --guidance-high 0.65 --cfg-scale 1.7
./eval_ckpts.sh --config configs/SIT-XL.yaml --exp-name SIT-XL --steps "1000000" --guidance-low 0.0 --guidance-high 0.65 --cfg-scale 1.8
