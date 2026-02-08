#!/usr/bin/env bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0

SPLIT='ranking'
RANKING_FILE='test_data_for_ranking_metric.jsonl'
RESULT_FILE="ranking_result.jsonl"
rm -f "${RESULT_FILE}"

TASK_MODEL_TYPES=(
    "gpt-4"
    "gemini-2.5-pro"
    "flan_t5"
    "llama3.1"
)

device=$1
model_name=$2   # t5-large / bart-large
task=$3         # ECQA / ESNLI
use_irm=$4
irm_penalty_weight=$5 # IRMv1 penalty weight (e.g., 0.1, 5, 10)
use_leak_probe=$6 # 1 or 0
leak_penalty_weight=$7 # leak probe penalty weight (e.g., 0.01, 0.001, 0.0005)
evaluate_on_task_model=$8
# 1: evaluate on task-model-generated rationales
# 0: evaluate on original human-annotated rationales

RANKING_TYPES=(
    gold
    gold_leaky
    vacuous
    leaky
    truncated_gold_80
    truncated_gold_50
    gold_noise
    shuffled_gold
)

if [ "$evaluate_on_task_model" -eq 1 ]; then
    for TASK_MODEL_NAME in "${TASK_MODEL_TYPES[@]}"; do
        echo "############################################"
        echo "Running task_model_type = ${TASK_MODEL_NAME}"
        echo "############################################"

        for ranking_type in "${RANKING_TYPES[@]}"; do
            echo "------------------------------"
            echo "ranking_type = ${ranking_type}"
            echo "------------------------------"

            python -m larev_eval \
                --task ${task} \
                --model_name ${model_name} \
                --split ${SPLIT} \
                --beams 2 \
                --device ${device} \
                --min_length 1 \
                --use_irm ${use_irm} \
                --irm_penalty_weight ${irm_penalty_weight} \
                --ranking_file ${RANKING_FILE} \
                --ranking_type ${ranking_type} \
                --use_leak_probe ${use_leak_probe} \
                --leak_penalty_weight ${leak_penalty_weight} \
                --ranking_result_file ${RESULT_FILE} \
                --task_model_type ${TASK_MODEL_NAME}
        done
    done
else
    echo "[INFO] Using original human-annotated rationales"

    for ranking_type in "${RANKING_TYPES[@]}"; do
        echo "------------------------------"
        echo "ranking_type = ${ranking_type}"
        echo "------------------------------"

        python -m larev_eval \
            --task ${task} \
            --model_name ${model_name} \
            --split ${SPLIT} \
            --beams 2 \
            --device ${device} \
            --min_length 1 \
            --use_irm ${use_irm} \
            --irm_penalty_weight ${irm_penalty_weight} \
            --ranking_file ${RANKING_FILE} \
            --ranking_type ${ranking_type} \
            --use_leak_probe ${use_leak_probe} \
            --leak_penalty_weight ${leak_penalty_weight} \
            --ranking_result_file ${RESULT_FILE} \
            --task_model_type None
    done
fi

