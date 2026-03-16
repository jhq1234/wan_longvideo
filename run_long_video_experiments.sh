#!/bin/bash

# Long Video Generation Experiments
# GPU 5번에서 30개의 프롬프트로 순차적으로 실험 진행

# Ctrl+C (SIGINT) 핸들러: 작업 중지 및 정리
cleanup_on_interrupt() {
    echo "" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
    echo "⚠️  Interrupted by user (Ctrl+C)" | tee -a "$LOG_FILE"
    echo "Stopping current process..." | tee -a "$LOG_FILE"
    echo "Interrupted at: $(date)" | tee -a "$LOG_FILE"
    echo "Last processed prompt: $prompt_num/30" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
    
    # 현재 실행 중인 Python 프로세스 그룹 종료
    # tee와 python 프로세스를 모두 종료하기 위해 프로세스 그룹 사용
    pkill -P $$ python 2>/dev/null
    
    exit 130  # SIGINT exit code
}

# SIGINT (Ctrl+C) 트랩 설정
trap cleanup_on_interrupt SIGINT

# 설정
GPU_ID=7
TASK="long-t2v-1.3B"
SIZE="832*480"
CKPT_DIR="/home/bispl_02/dohun/ReCamMaster_framepack_cdp_train_full/models/Wan-AI/Wan2.1-T2V-1.3B"
WINDOW_SIZE=81
MULTIPLIER=4
OVERLAP_START=41
LONG_STEPS=25
OFFLOAD_MODEL="False"

# 프롬프트 파일 경로
PROMPTS_FILE="prompts_30.txt"

# 출력 디렉토리
OUTPUT_DIR="./output_$(date +%Y%m%d)"
mkdir -p "$OUTPUT_DIR"

# 로그 파일
LOG_FILE="$OUTPUT_DIR/experiment_log.txt"

# Conda 초기화
source $(conda info --base)/etc/profile.d/conda.sh
conda activate wan

echo "==========================================" | tee -a "$LOG_FILE"
echo "Long Video Generation Experiments Started" | tee -a "$LOG_FILE"
echo "Date: $(date)" | tee -a "$LOG_FILE"
echo "GPU: $GPU_ID" | tee -a "$LOG_FILE"
echo "Total Prompts: $(wc -l < $PROMPTS_FILE)" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# 프롬프트 파일 읽기 및 순차 실행
prompt_num=1
while IFS= read -r prompt; do
    # 빈 줄 및 주석 건너뛰기
    if [[ -z "$prompt" ]] || [[ "$prompt" =~ ^# ]]; then
        continue
    fi
    
    echo "----------------------------------------" | tee -a "$LOG_FILE"
    echo "[$prompt_num/30] Processing prompt:" | tee -a "$LOG_FILE"
    echo "Prompt: $prompt" | tee -a "$LOG_FILE"
    echo "Start time: $(date)" | tee -a "$LOG_FILE"
    echo "----------------------------------------" | tee -a "$LOG_FILE"
    
    # 출력 파일명 생성 (프롬프트의 처음 50자를 사용)
    formatted_prompt=$(echo "$prompt" | sed 's/[^a-zA-Z0-9]/_/g' | cut -c1-50)
    timestamp=$(date +%Y%m%d_%H%M%S)
    size_formatted=$(echo "$SIZE" | tr '*' 'x')
    output_filename="${TASK}_${size_formatted}_1_1_${formatted_prompt}_${timestamp}.mp4"
    output_path="$OUTPUT_DIR/$output_filename"
    
    echo "Output file: $output_path" | tee -a "$LOG_FILE"
    
    # GPU 설정 및 실행
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    
    # Python 프로세스를 실행 (포그라운드)
    # Ctrl+C를 누르면 trap이 실행되어 정리 작업 수행
    python generate.py \
        --task "$TASK" \
        --size "$SIZE" \
        --ckpt_dir "$CKPT_DIR" \
        --long_window_size "$WINDOW_SIZE" \
        --long_multiplier "$MULTIPLIER" \
        --long_overlap_start "$OVERLAP_START" \
        --long_steps "$LONG_STEPS" \
        --offload_model "$OFFLOAD_MODEL" \
        --prompt "$prompt" \
        --save_file "$output_path" \
        2>&1 | tee -a "$LOG_FILE"
    
    exit_code=${PIPESTATUS[0]}
    
    echo "" | tee -a "$LOG_FILE"
    echo "End time: $(date)" | tee -a "$LOG_FILE"
    
    if [ $exit_code -eq 0 ]; then
        echo "✓ Successfully completed prompt $prompt_num" | tee -a "$LOG_FILE"
        echo "  Saved to: $output_path" | tee -a "$LOG_FILE"
    else
        echo "✗ Failed to process prompt $prompt_num (exit code: $exit_code)" | tee -a "$LOG_FILE"
        echo "Continuing with next prompt..." | tee -a "$LOG_FILE"
    fi
    
    echo "" | tee -a "$LOG_FILE"
    
    # 다음 프롬프트로
    ((prompt_num++))
    
    # GPU 메모리 정리를 위한 짧은 대기 (선택사항)
    sleep 2
    
done < "$PROMPTS_FILE"

echo "==========================================" | tee -a "$LOG_FILE"
echo "All Experiments Completed" | tee -a "$LOG_FILE"
echo "End time: $(date)" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
