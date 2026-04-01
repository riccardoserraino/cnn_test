#!/bin/bash
set -e

source venv/bin/activate

# Uncomment the following line if new pkg are needed
pip install -r requirements.txt

# to fix some terminal warnings related to tf
export TF_CPP_MIN_LOG_LEVEL=2 
export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_VISIBLE_DEVICES=""


python3 src/papi_main.py
