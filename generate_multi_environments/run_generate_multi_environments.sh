#!/bin/bash
set -e  # Exit on any error

# Default values
MODEL=${1:-"t5-large"}
TASK=${2:-"ECQA"}
SPLIT_TYPE=${3:-"train"}
GEMINI_KEY=${GEMINI_API_KEY:-""}
MODEL_NAME="gemini-2.5-flash-lite"


# Check if Gemini API key is provided
if [ -z "$GEMINI_KEY" ]; then
    echo "Error: GEMINI_API_KEY is not set."
    echo "Please export your Gemini API key before running this script."
    echo "Example:"
    echo "  export GEMINI_API_KEY=your_api_key_here"
    exit 1
fi

# Create output directory if it doesn't exist
OUTPUT_DIR="./output"
mkdir -p "$OUTPUT_DIR"

# Generate output filename
OUTPUT_FILE="$OUTPUT_DIR/multi_environments_${TASK}_${MODEL}_${SPLIT_TYPE}.jsonl"

echo "Starting multi environment generation..."
echo "Task: $TASK"
echo "Split type: $SPLIT_TYPE"
echo "Output file: $OUTPUT_FILE"
echo "Model: $MODEL_NAME"
echo ""

# Run the Python script
python generate_multi_environments.py \
    --task "$TASK" \
    --model_name_or_path "$MODEL" \
    --split_type "$SPLIT_TYPE" \
    --output "$OUTPUT_FILE" \
    --gemini_key "$GEMINI_KEY" \
    --model_name "$MODEL_NAME"

echo ""
echo "Multiple environment generation completed!"
echo "Output saved to: $OUTPUT_FILE"

# Optional: Show file size and line count
if [ -f "$OUTPUT_FILE" ]; then
    echo "File size: $(du -h "$OUTPUT_FILE" | cut -f1)"
    echo "Number of lines: $(wc -l < "$OUTPUT_FILE")"
fi
