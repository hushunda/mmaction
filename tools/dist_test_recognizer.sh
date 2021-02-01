#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}
#export CUDA_VISIBLE_DEVICES=5,4
$PYTHON -m torch.distributed.launch --nproc_per_node=$2 $(dirname "$0")/test_recognizer.py $1  --launcher pytorch $(@:3)
