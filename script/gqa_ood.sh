#!/usr/bin/env bash

delta=5

name=X_GGM_${delta}
output=snap/gqa_ood/$name
mkdir -p $output

# Training
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/gqa/gqa_ood.py \
    --train train \
    --valid val_tail \
    --llayers 9 \
    --xlayers 5 \
    --rlayers 5 \
    --loadLXMERTQA snap/pretrained/model \
    --gnn GCN \
    --num_layer 2 \
    --sigma 1.0 \
    --delta $delta \
    --numWorkers 8 \
    --tf_writer False \
    --bs 96 \
    --optim bert \
    --lr 5e-6 \
    --epochs 4 \
    --tqdm \
    --output $output ${@:3}
# Testing
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/gqa/gqa_ood.py \
    --train train \
    --valid ""  \
    --test testdev_all \
    --llayers 9 \
    --xlayers 5 \
    --rlayers 5 \
    --gnn GCN \
    --num_layer 2 \
    --numWorkers 8 \
    --tf_writer False \
    --load snap/gqa_ood/"$name"/BEST \
    --bs 512 \
    --optim bert \
    --tqdm \
    --output $output ${@:3}

