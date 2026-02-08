#!/usr/bin/env bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0

MODELS=("gpt-4" "gemini-2.5-pro" "flan_t5" "llama3.1")
GEMINI_KEY="${GEMINI_API_KEY:-}"
GPT_KEY="${GPT_API_KEY:-}"
LLAMA_TOKEN="${LLAMA_TOKEN:-}"

device=$1
task=$2  # ECQA / ESNLI

need_gemini=0
need_gpt=0
need_llama=0

for m in "${MODELS[@]}"; do
  case "$m" in
    gemini-* ) need_gemini=1 ;;
    gpt-4    ) need_gpt=1 ;;
    llama3.1 ) need_llama=1 ;;
  esac
done

if [ "$need_gemini" -eq 1 ] && [ -z "${GEMINI_KEY}" ]; then
  echo "Error: GEMINI_API_KEY is not set."
  echo "Please run: export GEMINI_API_KEY=your_api_key_here"
  exit 1
fi

if [ "$need_gpt" -eq 1 ] && [ -z "${GPT_KEY}" ]; then
  echo "Error: GPT_API_KEY is not set."
  echo "Please run: export GPT_API_KEY=your_api_key_here"
  exit 1
fi

if [ "$need_llama" -eq 1 ] && [ -z "${LLAMA_TOKEN}" ]; then
  echo "Error: LLAMA_TOKEN is not set."
  echo "Please run: export LLAMA_TOKEN=your_hf_token_here"
  exit 1
fi


for model in "${MODELS[@]}"; do
	OUTPUT_DIR="./output/${task}/${model}"
	mkdir -p "${OUTPUT_DIR}"
	OUTPUT_FILE="${OUTPUT_DIR}/generated_rationale.jsonl"
	echo "========================================"
	echo "Running task=${task}, model=${model} Writing to ${OUTPUT_FILE}"
	echo "========================================"

	python generate_rationale_from_task_model.py \
	--device ${device} \
	--task "${task}" \
	--model_name_or_path "${model}" \
	--gemini_key "${GEMINI_KEY}" \
	--gpt_key "${GPT_KEY}" \
	--llama_token "${LLAMA_TOKEN}" \
	--output_dir "${OUTPUT_FILE}"

	echo
done

