#!/usr/bin/env bash

CONFIG=$1          # TRT config (ex: bevformer_tiny_trt.py)
ENGINE=$2          # TensorRT engine (.plan / .engine)
GPU_ID=$3          # which GPU to use

LD_LIBRARY_PATH=/home/user/BEVFormer/tools/tensorRT:$LD_LIBRARY_PATH \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=$GPU_ID \
python $(dirname "$0")/test_trt.py \
    $CONFIG \
    $ENGINE