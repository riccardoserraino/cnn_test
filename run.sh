#!/bin/bash
set -e

source venv/bin/activate

pip install --default-timeout=1000 -q -r requirements.txt 

export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_VISIBLE_DEVICES=""
export TF_ENABLE_ONEDNN_OPTS=0

python3 src/cnn.py
