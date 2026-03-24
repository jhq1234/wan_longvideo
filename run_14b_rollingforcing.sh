#!/bin/bash

# long-t2v-14B experiments with prompts_rollingforcing.txt
# Same settings as launch.json "long-t2v-14B: western (GPU 3, offload)"

cleanup_on_interrupt() {
    echo ""
    echo "=========================================="
    echo "⚠️  Interrupted by user (Ctrl+C)"
    echo "Interrupted at: $(date)"
    echo "Last: prompt $prompt_num"
    echo "=========================================="
    pkill -P $$ python 2>/dev/null
    exit 130
}

trap cleanup_on_interrupt SIGINT

# 설정 (launch.json과 동일)
GPU_ID=5
TASK="long-t2v-14B"
SIZE="832*480"
CKPT_DIR="./models/Wan2.1-T2V-14B"
WINDOW_SIZE=81
MULTIPLIER=8
OVERLAP_START=41
LONG_STEPS=25
OFFLOAD_MODEL="True"
OVERLAP_MODE="both"

PROMPTS_FILE="prompts_rollingforcing.txt"

# 출력 디렉토리
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_OUTPUT_DIR="./output_14b_rollingforcing_${TIMESTAMP}"
mkdir -p "$BASE_OUTPUT_DIR"
LOG_FILE="$BASE_OUTPUT_DIR/experiment_log.txt"

# Conda 초기화
eval "$(conda shell.bash hook)"
conda activate wan

TOTAL_PROMPTS=$(grep -v '^$' "$PROMPTS_FILE" | grep -v '^#' | wc -l)

echo "==========================================" | tee -a "$LOG_FILE"
echo "14B Long Video Experiments (prompts_rollingforcing.txt)" | tee -a "$LOG_FILE"
echo "Date: $(date)" | tee -a "$LOG_FILE"
echo "GPU: $GPU_ID" | tee -a "$LOG_FILE"
echo "Prompts: $TOTAL_PROMPTS" | tee -a "$LOG_FILE"
echo "Output dir: $BASE_OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

export CUDA_VISIBLE_DEVICES=$GPU_ID

run_count=0
prompt_num=0

while IFS= read -r prompt; do
    if [[ -z "$prompt" ]] || [[ "$prompt" =~ ^# ]]; then
        continue
    fi

    ((prompt_num++))
    ((run_count++))

    PROMPT_DIR="${BASE_OUTPUT_DIR}/prompt_$(printf '%02d' $prompt_num)"
    mkdir -p "$PROMPT_DIR"
    OUTPUT_DIR="${PROMPT_DIR}"
    RUN_LOG="${OUTPUT_DIR}/log.txt"

    echo "----------------------------------------" | tee -a "$LOG_FILE"
    echo "[$run_count/$TOTAL_PROMPTS] Prompt $prompt_num" | tee -a "$LOG_FILE"
    echo "Prompt: ${prompt:0:80}..." | tee -a "$LOG_FILE"
    echo "Start time: $(date)" | tee -a "$LOG_FILE"
    echo "----------------------------------------" | tee -a "$LOG_FILE"

    python generate.py \
        --task "$TASK" \
        --size "$SIZE" \
        --ckpt_dir "$CKPT_DIR" \
        --long_window_size "$WINDOW_SIZE" \
        --long_multiplier "$MULTIPLIER" \
        --long_overlap_start "$OVERLAP_START" \
        --long_steps "$LONG_STEPS" \
        --offload_model "$OFFLOAD_MODEL" \
        --t5_cpu \
        --overlap_mode "$OVERLAP_MODE" \
        --sequential_windows true \
        --prompt "$prompt" \
        --save_file "${OUTPUT_DIR}/video.mp4" \
        2>&1 | tee "$RUN_LOG" | tee -a "$LOG_FILE"

    exit_code=${PIPESTATUS[0]}

    echo "" | tee -a "$LOG_FILE"
    echo "End time: $(date)" | tee -a "$LOG_FILE"
    if [ $exit_code -eq 0 ]; then
        echo "✓ Completed prompt_${prompt_num}" | tee -a "$LOG_FILE"
        echo "  Output: ${OUTPUT_DIR}/video.mp4" | tee -a "$LOG_FILE"
    else
        echo "✗ Failed prompt_${prompt_num} (exit code: $exit_code)" | tee -a "$LOG_FILE"
    fi
    echo "" | tee -a "$LOG_FILE"

    sleep 2

done < "$PROMPTS_FILE"

echo "==========================================" | tee -a "$LOG_FILE"
echo "All Experiments Completed" | tee -a "$LOG_FILE"
echo "End time: $(date)" | tee -a "$LOG_FILE"
echo "Results in: $BASE_OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "  Structure: prompt_NN/video.mp4" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
