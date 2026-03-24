#!/bin/bash

# Overlap Mode Experiments with prompts_rollingforcing.txt
# 각 프롬프트에 대해 3가지 overlap_mode (x0_weighted, velocity_interp, both) 실험

cleanup_on_interrupt() {
    echo ""
    echo "=========================================="
    echo "⚠️  Interrupted by user (Ctrl+C)"
    echo "Interrupted at: $(date)"
    echo "Last: prompt $prompt_num, overlap_mode $mode"
    echo "=========================================="
    pkill -P $$ python 2>/dev/null
    exit 130
}

trap cleanup_on_interrupt SIGINT

# 설정
GPU_IDS=(2 3)
MAX_PARALLEL=2
TASK="long-t2v-1.3B"
SIZE="832*480"
CKPT_DIR="./models/Wan2.1-T2V-1.3B"
WINDOW_SIZE=81
MULTIPLIER=12
OVERLAP_START=41
LONG_STEPS=25
OFFLOAD_MODEL="False"

PROMPTS_FILE="prompts_rollingforcing.txt"
OVERLAP_MODES=("x0_weighted" "velocity_interp" "both")

# 출력 디렉토리
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_OUTPUT_DIR="./output_overlap_rollingforcing_${TIMESTAMP}"
mkdir -p "$BASE_OUTPUT_DIR"
LOG_FILE="$BASE_OUTPUT_DIR/experiment_log.txt"

# Conda 초기화
eval "$(conda shell.bash hook)"
conda activate wan

TOTAL_PROMPTS=$(grep -v '^$' "$PROMPTS_FILE" | grep -v '^#' | wc -l)
TOTAL_RUNS=$((TOTAL_PROMPTS * 3))

echo "==========================================" | tee -a "$LOG_FILE"
echo "Overlap Mode Experiments (prompts_rollingforcing.txt)" | tee -a "$LOG_FILE"
echo "Date: $(date)" | tee -a "$LOG_FILE"
echo "GPUs: ${GPU_IDS[*]} (parallel: $MAX_PARALLEL)" | tee -a "$LOG_FILE"
echo "Prompts: $TOTAL_PROMPTS" | tee -a "$LOG_FILE"
echo "Total runs: $TOTAL_RUNS (${TOTAL_PROMPTS} prompts x 3 modes)" | tee -a "$LOG_FILE"
echo "Output dir: $BASE_OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

run_count=0
prompt_num=0
running_pids=()

run_single() {
    local gpu=$1
    local prompt_num=$2
    local mode=$3
    local prompt=$4
    local output_dir=$5
    local run_count=$6
    local total_runs=$7
    local run_log="${output_dir}/log.txt"

    (
        {
            echo "----------------------------------------"
            echo "[$run_count/$total_runs] Prompt $prompt_num, overlap_mode=$mode (GPU $gpu)"
            echo "Prompt: ${prompt:0:80}..."
            echo "Start time: $(date)"
            echo "----------------------------------------"

            CUDA_VISIBLE_DEVICES=$gpu python generate.py \
                --task "$TASK" \
                --size "$SIZE" \
                --ckpt_dir "$CKPT_DIR" \
                --long_window_size "$WINDOW_SIZE" \
                --long_multiplier "$MULTIPLIER" \
                --long_overlap_start "$OVERLAP_START" \
                --long_steps "$LONG_STEPS" \
                --offload_model "$OFFLOAD_MODEL" \
                --overlap_mode "$mode" \
                --prompt "$prompt" \
                --save_file "${output_dir}/video.mp4" \
                2>&1
        } | tee "$run_log" | tee -a "$LOG_FILE"
        exit_code=${PIPESTATUS[0]}

        {
            echo ""
            echo "End time: $(date)"
            if [ "${exit_code}" -eq 0 ]; then
                echo "✓ Completed prompt_${prompt_num} overlap_${mode} (GPU $gpu)"
                echo "  Output: ${output_dir}/video.mp4"
            else
                echo "✗ Failed prompt_${prompt_num} overlap_${mode} (GPU $gpu, exit code: $exit_code)"
            fi
            echo ""
        } | tee -a "$run_log" | tee -a "$LOG_FILE"

        exit "${exit_code}"
    )
}

while IFS= read -r prompt; do
    # 빈 줄 및 주석 건너뛰기
    if [[ -z "$prompt" ]] || [[ "$prompt" =~ ^# ]]; then
        continue
    fi

    ((prompt_num++))

    # 프롬프트별 서브디렉토리
    PROMPT_DIR="${BASE_OUTPUT_DIR}/prompt_$(printf '%02d' $prompt_num)"
    mkdir -p "$PROMPT_DIR"

    for i in "${!OVERLAP_MODES[@]}"; do
        mode="${OVERLAP_MODES[$i]}"
        ((run_count++))

        # 병렬 실행: 2개 이상 실행 중이면 하나 끝날 때까지 대기
        while [ ${#running_pids[@]} -ge $MAX_PARALLEL ]; do
            wait -n 2>/dev/null || true
            new_pids=()
            for p in "${running_pids[@]}"; do
                kill -0 "$p" 2>/dev/null && new_pids+=("$p")
            done
            running_pids=("${new_pids[@]}")
        done

        gpu=${GPU_IDS[$(( (run_count - 1) % MAX_PARALLEL ))]}
        OUTPUT_DIR="${PROMPT_DIR}/overlap_${mode}"
        mkdir -p "$OUTPUT_DIR"

        run_single "$gpu" "$prompt_num" "$mode" "$prompt" "$OUTPUT_DIR" "$run_count" "$TOTAL_RUNS" &
        pid=$!
        running_pids+=("$pid")

        sleep 1
    done

done < "$PROMPTS_FILE"

# 남은 작업 완료 대기
for pid in "${running_pids[@]}"; do
    wait "$pid" 2>/dev/null
done

echo "==========================================" | tee -a "$LOG_FILE"
echo "All Experiments Completed" | tee -a "$LOG_FILE"
echo "End time: $(date)" | tee -a "$LOG_FILE"
echo "Results in: $BASE_OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "  Structure: prompt_NN/overlap_{mode}/video.mp4" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
