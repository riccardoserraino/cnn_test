@echo off
call venv\Scripts\activate.bat
pip install --default-timeout=1000 -q -r requirements.txt 

set TF_CPP_MIN_LOG_LEVEL=3
set TF_ENABLE_ONEDNN_OPTS=0

python src\main.py