# Long Video Generation에서의 Temporal Consistency 문제 정의

## 문제 상황 요약

**현상**: Tweedie Caching 방법을 사용한 긴 비디오 생성 시, 각 chunk 내부에서 overlap 영역과 non-overlap 영역 사이의 경계에서 시각적 불일치(temporal inconsistency)가 발생합니다.

---

## 1. 문제의 정확한 기술적 정의

### 1.1 시스템 설정

- **방법**: Tweedie Caching (Sequential Window Processing with x₀ Caching)
- **Window 구조**: 
  - Window size: 81 frames
  - Multiplier: 4 (총 4개 window)
  - Overlap start: 41번째 프레임부터
  - 각 window는 순차적으로 처리됨

### 1.2 문제 발생 위치

**Chunk 2 (win_idx=1)를 예시로:**

```
Chunk 1: [========================================]
Chunk 2:              [========================================]
                      ↑                              ↑
                   Overlap                        Non-overlap
                   (재사용)                        (새로 생성)
```

**문제 발생 지점**: Chunk 2 내부에서 overlap 영역과 non-overlap 영역의 **경계**

### 1.3 문제의 구체적 메커니즘

**현재 구현 (문제 있는 코드):**

```python
# Chunk 2 처리 시:

# 1. 전체 window에 대해 모델이 velocity 예측 → x₀ 계산
window_x0 = window_latent - sigma * vp  # [C, T_win, H, W]

# 2. Overlap 영역: Chunk 1의 cached x₀로 완전히 교체
window_x0[:, reuse_start:reuse_end, :, :] = cached_x0_from_chunk1

# 3. Non-overlap 영역: Chunk 2가 새로 생성한 x₀ 유지
# window_x0[:, reuse_end:, :, :] = 원래 값 (Chunk 2가 생성한 것)
```

**문제점:**
- Overlap 영역: Chunk 1의 예측 결과 (이전 timestep의 정보)
- Non-overlap 영역: Chunk 2의 예측 결과 (현재 timestep의 정보)
- **두 영역이 서로 다른 "시점"의 정보를 담고 있어 경계에서 불연속성 발생**

---

## 2. 문제를 설명하는 다이어그램

### 2.1 Temporal Timeline 관점

```
Timestep t에서:

Chunk 1 (이미 완료):
  [===============] (x₀_1 계산 완료, 캐시에 저장)
  
Chunk 2 (현재 처리 중):
  [===============] (전체 window에 대해 x₀_2 계산)
    ↑              ↑
  Overlap      Non-overlap
  (x₀_1 사용)  (x₀_2 사용)
  
문제: x₀_1과 x₀_2가 서로 다른 "denoising history"를 가지고 있음
```

### 2.2 Information Flow 관점

```
Chunk 1의 정보 흐름:
  x_t → model → v_pred → x₀_1 → (캐시 저장)
  
Chunk 2의 정보 흐름:
  Overlap:     x_t → (skip) → x₀_1 (캐시에서 가져옴)
  Non-overlap: x_t → model → v_pred → x₀_2 (새로 계산)
  
문제: 같은 x_t에서 시작했지만, 두 경로가 다른 결과를 만들어냄
```

---

## 3. 문제의 수학적 표현

### 3.1 현재 구현

각 timestep `t`에서:

**Chunk 2의 x₀ 구성:**

```
x₀_chunk2 = {
  x₀_chunk2[0:reuse_end] = x₀_chunk1[overlap]  (cached, from chunk 1)
  x₀_chunk2[reuse_end:] = x₀_chunk2[reuse_end:]  (newly computed, from chunk 2)
}
```

**문제의 수학적 표현:**

```
x₀_chunk2[reuse_end - 1] ≠ x₀_chunk2[reuse_end]
```

왜냐하면:
- `x₀_chunk2[reuse_end - 1]`은 Chunk 1의 denoising history를 따름
- `x₀_chunk2[reuse_end]`는 Chunk 2의 denoising history를 따름
- 두 history가 다르므로 불연속성 발생

### 3.2 이상적인 상황

```
x₀_chunk2[reuse_end - 1] ≈ x₀_chunk2[reuse_end]
```

이를 위해서는:
- Overlap 영역도 Chunk 2의 모델 예측을 일부 반영해야 함
- 또는 두 영역 사이의 경계를 부드럽게 전환해야 함

---

## 4. 문제를 설명하는 비유

### 비유 1: 두 명의 화가가 같은 그림을 그리는 경우

- **Chunk 1**: 화가 A가 전체 그림의 왼쪽 절반을 그림
- **Chunk 2**: 화가 B가 전체 그림의 오른쪽 절반을 그림
  - 왼쪽 절반: 화가 A가 그린 것을 그대로 사용
  - 오른쪽 절반: 화가 B가 새로 그림

**문제**: 두 화가의 스타일이 다르면 경계에서 어색함이 발생

**해결책**: 경계 부분을 두 화가가 협력해서 그리기 (blending)

### 비유 2: 두 개의 비디오 클립을 이어붙이는 경우

- **Chunk 1**: 첫 번째 클립 (이미 완성)
- **Chunk 2**: 두 번째 클립 (현재 편집 중)
  - 앞부분: 첫 번째 클립의 끝부분을 그대로 사용
  - 뒷부분: 두 번째 클립을 새로 촬영

**문제**: 두 클립의 색감, 조명, 움직임이 다르면 경계에서 어색함

**해결책**: Cross-fade 또는 color grading으로 자연스럽게 전환

---

## 5. 문제를 설명할 때 포함해야 할 핵심 정보

### 필수 정보:

1. **방법**: Tweedie Caching (x₀ caching)
2. **문제 발생 위치**: 각 chunk 내부의 overlap/non-overlap 경계
3. **문제의 원인**: 
   - Overlap 영역: 이전 chunk의 cached x₀ (hard replacement)
   - Non-overlap 영역: 현재 chunk의 새로 생성된 x₀
   - 두 영역이 서로 다른 denoising history를 가짐
4. **시각적 증상**: 경계에서 색감, 움직임, 구조의 불연속성

### 선택적 정보:

- Window 크기, overlap 크기 등 구체적 파라미터
- 발생하는 timestep (초반/후반)
- 특정 프롬프트나 장면에서 더 두드러지는지

---

## 6. 문제 설명 예시 (다른 사람에게 설명할 때)

### 짧은 버전 (1-2문장):

"Tweedie Caching 방법에서 각 chunk의 overlap 영역은 이전 chunk의 cached x₀를 사용하고, non-overlap 영역은 새로 생성된 x₀를 사용하는데, 이 두 영역 사이의 경계에서 temporal consistency가 깨집니다."

### 중간 버전 (단락):

"Tweedie Caching 방법을 사용한 긴 비디오 생성에서 문제가 발생합니다. 각 chunk(예: Chunk 2)를 처리할 때, overlap 영역(첫 번째 절반)은 Chunk 1에서 계산된 cached x₀를 그대로 재사용하고, non-overlap 영역(두 번째 절반)은 Chunk 2에서 새로 계산된 x₀를 사용합니다. 문제는 이 두 영역이 서로 다른 denoising history를 가지고 있어서, 경계에서 시각적 불일치(temporal inconsistency)가 발생한다는 것입니다. 예를 들어, Chunk 2의 40번째 프레임(overlap 끝)과 41번째 프레임(non-overlap 시작) 사이에서 색감이나 움직임이 급격히 변하는 현상이 관찰됩니다."

### 긴 버전 (상세 설명):

"긴 비디오 생성을 위해 Tweedie Caching 방법을 사용하고 있습니다. 이 방법은 각 temporal window를 순차적으로 처리하면서, overlap 영역의 x₀를 캐시에 저장하고 다음 window에서 재사용합니다.

구체적으로, Chunk 2를 처리하는 과정을 살펴보면:

1. **초기화**: Chunk 2의 전체 window에 대해 random noise 생성
2. **Overlap 영역 처리**: Chunk 1의 최종 denoised latent에 noise를 재주입
3. **Denoising**: 각 timestep에서:
   - 전체 window에 대해 모델이 velocity 예측 → x₀ 계산
   - **Overlap 영역**: Chunk 1의 cached x₀로 완전히 교체 (hard replacement)
   - **Non-overlap 영역**: Chunk 2가 새로 계산한 x₀ 유지

문제는 overlap 영역과 non-overlap 영역이 서로 다른 'denoising trajectory'를 따르기 때문에, 경계에서 불연속성이 발생한다는 것입니다. Overlap 영역은 Chunk 1의 denoising history를 따르고, non-overlap 영역은 Chunk 2의 denoising history를 따르는데, 이 두 history가 일치하지 않아서 경계에서 시각적 어색함이 발생합니다.

이 문제를 해결하기 위해서는 overlap 영역에서도 Chunk 2의 모델 예측을 일부 반영하거나, 두 영역 사이의 경계를 부드럽게 전환하는 방법이 필요합니다."

---

## 7. 문제 해결 방향 제시

### 해결책 1: Velocity Interpolation

Overlap 영역에서:
- Chunk 1의 cached velocity와 Chunk 2의 모델 velocity를 interpolate
- Interpolated velocity로 x₀ 계산
- 이렇게 하면 Chunk 2의 모델 예측도 일부 반영됨

### 해결책 2: x₀ Blending

Overlap 영역에서:
- Chunk 1의 cached x₀와 Chunk 2의 새로 계산된 x₀를 blend
- 경계에서 부드러운 전환

### 해결책 3: Gradual Transition Zone

Overlap 영역 내에서:
- 경계 근처에서 점진적으로 전환
- Hard boundary 대신 soft transition zone 사용

---

## 8. 검증 가능한 가설

**가설**: Velocity interpolation을 사용하면 overlap 영역에서도 현재 chunk의 모델 예측을 반영하여, overlap/non-overlap 경계의 일관성이 개선될 것이다.

**검증 방법**:
- 같은 설정에서 velocity interpolation 사용/미사용 비교
- 경계 영역의 temporal consistency metric 측정
- 시각적 품질 평가
