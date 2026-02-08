#!/usr/bin/env bash
# env definf
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0

device=$1
model=$2 # t5-large / bart-large
task=$3 # ECQA / ESNLI

OUT_DIR="./output"
MASK_TOKEN="<mask>"
MODEL_DIR="./output/${task}_temp-${model}" # Path to the fine-tuned model which will be used to compute IG

python compute_ig.py \
	--device ${device} \
        --task ${task} \
        --model_name_or_path ${model} \
        --model_dir "${MODEL_DIR}" \
        --device ${device} \
        --mask_token ${MASK_TOKEN}
