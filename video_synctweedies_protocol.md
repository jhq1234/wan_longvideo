# Video SyncTweedies: Flow Model x₀ Weighted Averaging for Long Video Generation

## 목적

이 문서는 Flow Matching 기반 diffusion model에서  
**predicted x₀ weighted averaging**을 이용해 긴 비디오를 생성하는 기법의  
핵심 아이디어와 구현 시 중요하게 고려해야 할 사항들을 정리한 것입니다.

새로운 프로젝트(Wan 등 video diffusion model)에서 이 기법을 구현하는  
LLM 에이전트를 위한 참고 문서입니다.

---

## 1. 핵심 아이디어

### 문제 정의

비디오 모델은 한 번에 처리할 수 있는 프레임 수에 제한이 있습니다.  
이 제약 안에서 제한보다 **긴 비디오**를 일관성 있게 생성하는 것이 목표입니다.

### 핵심 전략: Overlapping Window + Weighted Averaging

전체 비디오를 overlapping temporal window 단위로 나누어 denoising하고,  
overlap 구간에서 여러 window의 predicted x₀를 **단순 평균**합니다.

```
전체 영상:  [========================]
Window A:   [=======]
Window B:       [=======]
Window C:           [=======]
Overlap:        [===][===]
                  ↑
              이 구간의 x₀를 A, B 두 window의 예측값으로 평균
```

### Flow Matching에서의 x₀ 예측

Flow Matching (SD3, Wan, CogVideoX 등)의 Tweedie 공식:

```
x̂₀ = x_t - σ_t * v_pred       (σ_t = t / 1000)
```

매 denoising step에서 모델은 각 window에 대해 v_pred를 예측하고,  
이로부터 x̂₀를 계산합니다. 이 x̂₀를 overlap 구간에서 평균하여  
전체 영상의 현재 상태를 업데이트합니다.

---

## 2. 전체 알고리즘 흐름

```
초기화:
  video_latent = random_noise  (전체 영상 크기의 latent 버퍼)

for step in range(max_steps):

  [A] 현재 video_latent에서 overlapping window들을 추출
      → latent_windows  [num_windows, C, T_win, H, W]

  [B] 시간 t 샘플링 (max → min으로 annealing)

  [C] 각 window에 노이즈 추가
      latent_noisy = (1 - σ_t) * latent_windows + σ_t * ε

  [D] ODE로 denoising: latent_noisy → x̂₀
      각 window를 독립적으로 t → 0까지 denoising
      x0_pred  [num_windows, C, T_win, H, W]

  [E] video_latent 업데이트 (아래 두 방식 중 선택):

      [평균 모드]   video_latent[t : t+T_win] = mean of all x0_pred covering t
      [개별 모드]   video_latent[t : t+T_win] = x0_pred[window_i]  (마지막 write 우선)
```

---

## 3. 구현 시 중요하게 고려한 것들

### 3.1 Latent space에서 averaging

averaging은 반드시 **latent space**에서 수행해야 합니다.  
pixel space에서 averaging하면 VAE encode/decode를 반복해야 하므로  
품질 저하와 속도 문제가 생깁니다.

즉, video_latent 버퍼도 latent space에서 유지하고,  
x̂₀도 latent 형태로 받아서 직접 평균합니다.

### 3.2 Coverage 보장

averaging 시 **모든 temporal position이 최소 1개 이상의 window에 커버**되어야 합니다.  
linspace로 window의 시작점을 균등 배치하거나,  
stride ≤ window_len 조건을 명시적으로 검증하는 것이 중요합니다.  
커버되지 않은 구간이 생기면 그 부분은 그냥 random noise로 남게 됩니다.

### 3.3 Gradient 없이 closed-form 업데이트

기존 diffusion model에서 SDS처럼 gradient descent로 업데이트하는 방식과 달리,  
이 기법은 gradient를 전혀 사용하지 않습니다.  
x̂₀를 직접 video_latent에 할당(평균)하는 **closed-form update**입니다.

따라서 `torch.no_grad()` 컨텍스트 안에서 모든 denoising이 이루어지고,  
업데이트 시 `.data`를 통한 in-place 쓰기 또는 새 tensor 할당으로 처리합니다.

### 3.4 평균 중단 타이밍 (avg_ratio / stop_step)

실험적으로 발견한 것:

- **초반 step**: t가 커서 x̂₀ 예측이 거칠지만, 이 시점의 averaging이 전체 영상의 색감·구조 일관성을 잡아줌
- **후반 step**: t가 작아져 x̂₀가 세밀해지는 단계에서 averaging을 계속하면 window 경계가 blurry해질 수 있음

이를 제어하기 위해 두 가지 스케줄:

```
Forward (avg → individual):
  step < stop_step:  weighted averaging
  step ≥ stop_step:  각 window 독립 업데이트

Reverse (individual → avg):
  step < start_step: 각 window 독립 업데이트
  step ≥ start_step: weighted averaging
```

`stop_step = max_steps * ratio` 형태로 ratio (0.0~1.0)를 제어 파라미터로 씁니다.

### 3.5 Individual update 시 ordering 주의

개별 모드에서는 마지막 window의 write가 이전 것을 덮어씁니다.  
window를 순서대로 iterate하면 뒤쪽 window가 overlap 구간을 지배하게 됩니다.  
이것이 문제라면 averaging 없이도 **random ordering** 또는  
**alternating (홀짝) ordering**으로 편향을 줄일 수 있습니다.

### 3.6 ODE denoising은 각 window를 독립적으로

전체 영상을 한 번에 ODE denoising하는 것이 아니라,  
**각 window를 독립적으로** `t → 0`까지 denoising합니다.  
따라서 batch dimension에 window들을 쌓아서 한 번의 forward pass로 처리합니다:

```
input:  [num_windows, C, T_win, H, W]  → 모델 batch로 처리
output: [num_windows, C, T_win, H, W]  → 각 window의 x̂₀
```

메모리가 부족하면 window를 나눠서 처리합니다.

### 3.7 Time annealing

t를 step에 따라 **linear하게 감소**시킵니다 (t_max → t_min):

- 초반(t 큼, σ 큼): x̂₀가 rough하지만 전역 구조를 결정
- 후반(t 작음, σ 작음): x̂₀가 세밀해지며 texture 완성

t_min을 0이 아닌 값(예: 200)으로 설정하면  
마지막에 완전히 denoising되지 않아 지나친 sharpness를 방지합니다.

---

## 4. 구현 체크리스트 (모델 무관)

```
[ ] 전체 video latent 버퍼를 latent space에서 유지
[ ] temporal window sampling: linspace로 균등 분포, coverage 100% 보장
[ ] 각 window를 batch로 묶어 한 번에 ODE denoising
[ ] closed_form_optimize: latent space에서 weighted average, gradient 없음
[ ] individual_optimize: in-place 쓰기, .data 또는 detach() 사용
[ ] closed_form_stop_step 파라미터: forward averaging schedule 제어
[ ] closed_form_start_step 파라미터: reverse averaging schedule 제어
[ ] t annealing: linear, t_max ≈ 999 → t_min ≈ 200
[ ] 최종 output: video latent를 VAE decode하여 mp4 저장
```

---

## 5. 한 줄 요약

> 매 step에서 **각 temporal window를 독립적으로 denoising**해 x̂₀를 얻고,  
> **overlap 구간에서 평균**하여 전체 video latent를 업데이트한다.  
> Gradient descent 없이 closed-form으로 동작하며,  
> averaging ↔ individual 전환 타이밍을 sweep하여 최적값을 찾는다.
