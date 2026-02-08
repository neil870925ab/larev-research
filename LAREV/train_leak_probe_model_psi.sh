#!/usr/bin/env bash
# env definf
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0

device=$1
model=$2 # t5-large / bart-large
task=$3 # ECQA / ESNLI
epochs=$4 # set it to 4 to reproduce the results in the thesis.
lr=$5 #set it to 3e-5 to reproduce the results in the thesis.

OUT_DIR="./output"
OUT_NAME="psi_model_weight.bin"

python train_leak_probe_model_psi.py \
        --task ${task} \
        --out_dir "${OUT_DIR}/${task}-${model}_leak_probe_model" \
        --out_name ${OUT_NAME} \
        --model_name_or_path ${model} \
        --device ${device} \
        --num_train_epochs ${epochs} \
        --learning_rate ${lr} \
        --max_input_length 300 \
        --max_length 20 \
        --train_batch_size 16 \
        --eval_batch_size 16 \
        --phi_model_path "${OUT_DIR}/${task}_regular-${model}" \
