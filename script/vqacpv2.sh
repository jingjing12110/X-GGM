#!/usr/bin/env bash

delta=8

name=X-GGM_${delta}
output=snap/vqa/$name
mkdir -p $output

# Training
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/vqa/vqacpv2.py \
    --train train \
    --valid dev_test  \
    --llayers 9 \
    --xlayers 5 \
    --rlayers 5 \
    --loadLXMERTQA snap/pretrained/model \
    --numWorkers 4 \
    --gnn GCN \
    --num_layer 2 \
    --sigma 1.0 \
    --delta 0 \
    --bs 92 \
    --optim bert \
    --lr 1e-6 \
    --epochs 4 \
    --tf_writer False \
    --tqdm \
    --output $output ${@:3}


# Testing OOD
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/vqa/vqacpv2.py \
    --train train \
    --valid "" \
    --test test \
    --tmode OOD \
    --llayers 9 \
    --xlayers 5 \
    --rlayers 5 \
    --numWorkers 8 \
    --gnn GCN \
    --num_layer 2 \
    --load snap/final/"$name"/BEST \
    --tf_writer False \
    --bs 512\
    --optim bert \
    --tqdm \
    --output $output ${@:3}

# Testing ID
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/vqa/vqacpv2.py \
    --train train \
    --valid "" \
    --test val \
    --tmode ID \
    --llayers 9 \
    --xlayers 5 \
    --rlayers 5 \
    --numWorkers 8 \
    --gnn GCN \
    --num_layer 2 \
    --load snap/final/"$name"/BEST \
    --tf_writer False \
    --bs 512 \
    --optim bert \
    --tqdm \
    --output $output ${@:3}
