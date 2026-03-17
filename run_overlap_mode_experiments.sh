#!/bin/bash

# Overlap Mode Experiments: 4가지 경우의 수 (x0_weighted, velocity_interp, both, last_write)
# launch.json 설정 기반

cleanup_on_interrupt() {
    echo ""
    echo "=========================================="
    echo "⚠️  Interrupted by user (Ctrl+C)"
    echo "Interrupted at: $(date)"
    echo "=========================================="
    pkill -P $$ python 2>/dev/null
    exit 130
}

trap cleanup_on_interrupt SIGINT

# launch.json과 동일한 설정
GPU_ID=3
TASK="long-t2v-1.3B"
SIZE="832*480"
CKPT_DIR="./models/Wan2.1-T2V-1.3B"
WINDOW_SIZE=81
MULTIPLIER=12
OVERLAP_START=41
LONG_STEPS=25
OFFLOAD_MODEL="False"

PROMPT="Late-1960s revisionist western shot on faded 35mm film with grain and subtle analog texture. A man gallops across an open plain carrying a large American flag as wind and dust trail behind the horse. Side tracking shot from a moving vehicle with slight vibration and natural in-camera motion blur under a muted cloudy sky."

# 4가지 overlap_mode
OVERLAP_MODES=("x0_weighted" "velocity_interp" "both" "last_write")

# 출력 디렉토리 (타임스탬프 포함)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_OUTPUT_DIR="./output_overlap_mode_${TIMESTAMP}"
mkdir -p "$BASE_OUTPUT_DIR"
LOG_FILE="$BASE_OUTPUT_DIR/experiment_log.txt"

# Conda 초기화
eval "$(conda shell.bash hook)"
conda activate wan

echo "==========================================" | tee -a "$LOG_FILE"
echo "Overlap Mode Experiments (4 cases)" | tee -a "$LOG_FILE"
echo "Date: $(date)" | tee -a "$LOG_FILE"
echo "GPU: $GPU_ID" | tee -a "$LOG_FILE"
echo "Output dir: $BASE_OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

export CUDA_VISIBLE_DEVICES=$GPU_ID

for i in "${!OVERLAP_MODES[@]}"; do
    mode="${OVERLAP_MODES[$i]}"
    mode_num=$((i + 1))
    
    echo "----------------------------------------" | tee -a "$LOG_FILE"
    echo "[$mode_num/4] overlap_mode=$mode" | tee -a "$LOG_FILE"
    echo "Start time: $(date)" | tee -a "$LOG_FILE"
    echo "----------------------------------------" | tee -a "$LOG_FILE"
    
    # 각 모드별 서브디렉토리
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/overlap_${mode}"
    mkdir -p "$OUTPUT_DIR"
    
    python generate.py \
        --task "$TASK" \
        --size "$SIZE" \
        --ckpt_dir "$CKPT_DIR" \
        --long_window_size "$WINDOW_SIZE" \
        --long_multiplier "$MULTIPLIER" \
        --long_overlap_start "$OVERLAP_START" \
        --long_steps "$LONG_STEPS" \
        --offload_model "$OFFLOAD_MODEL" \
        --overlap_mode "$mode" \
        --prompt "$PROMPT" \
        --save_file "${OUTPUT_DIR}/video.mp4" \
        2>&1 | tee -a "$LOG_FILE"
    
    exit_code=${PIPESTATUS[0]}
    
    echo "" | tee -a "$LOG_FILE"
    echo "End time: $(date)" | tee -a "$LOG_FILE"
    
    if [ $exit_code -eq 0 ]; then
        echo "✓ Completed overlap_mode=$mode" | tee -a "$LOG_FILE"
        echo "  Output: ${OUTPUT_DIR}/video.mp4" | tee -a "$LOG_FILE"
    else
        echo "✗ Failed overlap_mode=$mode (exit code: $exit_code)" | tee -a "$LOG_FILE"
    fi
    
    echo "" | tee -a "$LOG_FILE"
    
    # 다음 실험 전 짧은 대기
    sleep 2
done

echo "==========================================" | tee -a "$LOG_FILE"
echo "All 4 Overlap Mode Experiments Completed" | tee -a "$LOG_FILE"
echo "End time: $(date)" | tee -a "$LOG_FILE"
echo "Results in: $BASE_OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "  - overlap_x0_weighted/video.mp4" | tee -a "$LOG_FILE"
echo "  - overlap_velocity_interp/video.mp4" | tee -a "$LOG_FILE"
echo "  - overlap_both/video.mp4" | tee -a "$LOG_FILE"
echo "  - overlap_last_write/video.mp4" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
