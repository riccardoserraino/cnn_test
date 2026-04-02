#!/bin/bash
set -e

source venv/bin/activate
pip install --default-timeout=1000 -r requirements.txt

export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=""

python3 src/papi_main.py
