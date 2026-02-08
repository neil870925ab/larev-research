#!/usr/bin/env bash
# env definf
# Training script for phi and phi base model
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0

device=$1
data_type=$2 # regular (r, b) / temp (b)
model=$3 # t5-large / bart-large
task=$4 # ECQA / ESNLI 
epochs=$5 # set it to 8 to reproduce the results in the thesis.
lr=$6 # set it to 3e-5 to reproduce the results in the thesis.
use_leak_probe=$7 # 1 or 0
leak_penalty_weight=$8 # leak probe penalty weight (e.g., 0.01, 0.001, 0.0005)

PSI_MODEL_PATH="./output/${task}-${model}_leak_probe_model/psi_model_weight.bin"

avoid_dot_in_filename=$(echo ${leak_penalty_weight} | sed 's/\./p/g')

if [ "$use_leak_probe" -eq 1 ]; then
    OUT_DIR="./output/${task}_${data_type}-${model}-leak_probe_penalty_${avoid_dot_in_filename}"
else
    OUT_DIR="./output/${task}_${data_type}-${model}"
fi

python larev_train.py \
        --task ${task} \
        --data_type ${data_type} \
        --out_dir ${OUT_DIR} \
        --model_name_or_path ${model} \
        --device ${device} \
        --num_train_epochs ${epochs} \
        --learning_rate ${lr} \
        --do_train \
        --do_eval \
        --eval_during_train \
        --save_total_limit 1 \
        --overwrite_cache \
        --max_input_length 300 \
        --min_length 1 \
        --max_length 20 \
        --logging_steps 32 \
        --gradient_accumulation_steps 8 \
        --train_batch_size 8 \
        --eval_batch_size 8 \
        --overwrite_out_dir \
        --beams 2 \
        --use_leak_probe ${use_leak_probe} \
        --psi_model_path ${PSI_MODEL_PATH} \
        --leak_penalty_weight ${leak_penalty_weight}

