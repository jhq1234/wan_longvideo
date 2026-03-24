#!/bin/bash

# GPU Placeholder Script
# GPU 2, 3번을 계속 사용하되 데이터는 저장하지 않음
# GPU를 점유하여 다른 작업이 사용하지 못하도록 함

# Ctrl+C 핸들러
cleanup_on_interrupt() {
    echo ""
    echo "=========================================="
    echo "⚠️  Stopping GPU placeholder..."
    echo "Interrupted at: $(date)"
    echo "=========================================="
    pkill -P $$ python 2>/dev/null
    exit 130
}

trap cleanup_on_interrupt SIGINT

# Conda 초기화
eval "$(conda shell.bash hook)"
conda activate wan

echo "=========================================="
echo "GPU Placeholder Started"
echo "Date: $(date)"
echo "GPUs: 2, 3"
echo "Mode: Continuous computation (no data saving)"
echo "=========================================="
echo ""

# GPU 2번용 placeholder (백그라운드)
# 약 24GB 메모리 점유 + 적절한 GPU 사용률 (무리 없이)
export CUDA_VISIBLE_DEVICES=2
python -c "
import torch
print('GPU 2: Starting placeholder computation (target: ~24GB, 50-60% utilization)...')
# 24GB 메모리 점유를 위한 큰 텐서 할당
# float32: 4 bytes per element
# 16384 x 16384 = 268,435,456 elements = ~1GB per tensor
# 24GB를 위해 약 24개 할당
tensors = []
for i in range(24):
    t = torch.randn(16384, 16384, device='cuda', dtype=torch.float32)
    tensors.append(t)
    if (i+1) % 5 == 0:
        print(f'GPU 2: Allocated {i+1}/24 tensors, ~{torch.cuda.memory_allocated()/1024**3:.2f}GB')
print(f'GPU 2: Final allocation: ~{torch.cuda.memory_allocated()/1024**3:.2f}GB')
print('GPU 2: Starting continuous computation (50-60% utilization, ~24GB memory)...')
# GPU 사용률을 50-60%로 조절 (연산:휴식 ≈ 1:2)
import time
while True:
    # 최소한의 연산만 수행 (메모리 유지용)
    result = torch.matmul(tensors[0], tensors[1])
    del result
    torch.cuda.synchronize()  # GPU 연산 완료 대기
    time.sleep(0.5)  # 휴식 시간을 더 늘려서 GPU 사용률을 50-60%로 조절
" &
GPU2_PID=$!

# GPU 3번용 placeholder (백그라운드)
# 약 24GB 메모리 점유 + 적절한 GPU 사용률 (무리 없이)
export CUDA_VISIBLE_DEVICES=3
python -c "
import torch
print('GPU 3: Starting placeholder computation (target: ~24GB, 50-60% utilization)...')
# 24GB 메모리 점유를 위한 큰 텐서 할당
tensors = []
for i in range(24):
    t = torch.randn(16384, 16384, device='cuda', dtype=torch.float32)
    tensors.append(t)
    if (i+1) % 5 == 0:
        print(f'GPU 3: Allocated {i+1}/24 tensors, ~{torch.cuda.memory_allocated()/1024**3:.2f}GB')
print(f'GPU 3: Final allocation: ~{torch.cuda.memory_allocated()/1024**3:.2f}GB')
print('GPU 3: Starting continuous computation (50-60% utilization, ~24GB memory)...')
# GPU 사용률을 50-60%로 조절 (연산:휴식 ≈ 1:2)
import time
while True:
    # 최소한의 연산만 수행 (메모리 유지용)
    result = torch.matmul(tensors[0], tensors[1])
    del result
    torch.cuda.synchronize()  # GPU 연산 완료 대기
    time.sleep(0.5)  # 휴식 시간을 더 늘려서 GPU 사용률을 50-60%로 조절
" &
GPU3_PID=$!

echo "GPU 2 placeholder PID: $GPU2_PID"
echo "GPU 3 placeholder PID: $GPU3_PID"
echo ""
echo "GPUs are now occupied. Press Ctrl+C to stop."
echo ""

# 메인 프로세스는 대기
wait
