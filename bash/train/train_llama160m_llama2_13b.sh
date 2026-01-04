#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
/scratch/yf3005/rosetta/bin/torchrun --nproc_per_node=1 --master_port=29502 script/train/SFT_train.py \
    --config recipe/train_recipe/C2C_llama160m+llama2_13b.json \
