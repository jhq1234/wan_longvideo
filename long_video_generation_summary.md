# Wan Video 생성 방식 및 Long Video 실험 정리

## 1) 현재 Wan Video 생성 파이프라인 (코드 기준)

메인 엔트리포인트는 `generate.py`이며, task별로 서로 다른 파이프라인 클래스를 호출한다.

- `t2v` / `t2i`: `wan.WanT2V`
- `i2v`: `wan.WanI2V`
- `flf2v`: `wan.WanFLF2V`
- `vace`: `wan.WanVace`
- `long-t2v-*`: `wan.WanT2VLong` (`wan/long_video.py`)

공통 흐름:
1. argparse로 설정 파싱
2. task/size/frame/seed 검증
3. (옵션) prompt extension
4. 파이프라인 객체 생성
5. 샘플링 수행
6. 결과 저장 (`cache_video` / `cache_image`)

---

## 2) Long Video 기본 아이디어 (현재 메인 코드)

`wan/long_video.py`의 `WanT2VLong.generate_long()`은 SyncTweedies 기반으로 동작한다.

핵심 개념:
- 모델 native 길이(`window_size`, 보통 81)로 여러 청크를 생성
- 청크 간 overlap 구간을 정의(`overlap_start`)
- 각 timestep에서 청크별 예측을 전역 latent로 합성
- 합성된 `x0_full`로 UniPC step 진행

주요 파라미터:
- `window_size`: 청크 길이 (4n+1 제약)
- `multiplier`: 청크 개수
- `overlap_start`: 청크 내부 overlap 시작 프레임
- `long_steps`: 외부 denoising step 수

총 프레임 계산(픽셀 프레임):
- `total_frames = window_size + (multiplier - 1) * (window_size - overlap_start)`

---

## 3) Long Video에서 시도한 overlap 합성 방식

현재 메인 코드(`generate.py` + `wan/long_video.py`) 기준으로 `--overlap_mode`가 4가지다.

### (A) `x0_weighted`
- 방식: overlap 구간에서 청크별 `x0`를 선형 가중 평균
- 구현 함수: `_aggregate_x0_weighted()`
- 의미: 기본 SyncTweedies 합성 방식

### (B) `velocity_interp`
- 방식: overlap 구간에서 velocity(`vp`)를 먼저 보간하고, `x0 = x_t - sigma * vp_blended` 계산
- 구현 함수: `_aggregate_x0_weighted_velocity_interp()`
- 의미: `x0` 직접 평균 대신 velocity field를 부드럽게 연결

### (C) `both`
- 방식: `x0_weighted` 결과와 `velocity_interp` 결과를 평균
- 구현 함수: `_aggregate_x0_both()`
- 수식: `x0_final = (x0_weighted + x0_velocity_interp) / 2`

### (D) `last_write`
- 방식: overlap에서 평균 없이 뒤 청크 값으로 덮어쓰기
- 구현 함수: `_aggregate_x0(..., avg_mode=False)`
- 의미: blending 없는 hard overwrite 기준선

참고:
- `--velocity_interpolation` 플래그는 사실상 `--overlap_mode velocity_interp` 동작으로 연결된다.

---

## 4) 실험 자동화 스크립트

### 4-1) 단일 프롬프트 4모드 비교
- 파일: `run_overlap_mode_experiments.sh`
- 모드: `x0_weighted`, `velocity_interp`, `both`, `last_write`
- 목적: 동일 프롬프트에서 overlap 합성 방식 비교

### 4-2) rollingforcing 프롬프트 배치 실험
- 파일: `run_overlap_mode_rollingforcing.sh`
- 프롬프트 파일: `prompts_rollingforcing.txt`
- 현재 모드: `x0_weighted`, `velocity_interp`, `both` (`last_write` 제외)
- 목적: 여러 프롬프트에서 3가지 모드 반복 비교

---

## 5) 과거/백업 라인에서 시도한 방식 (참고)

`generate_backup0317.py` + `wan/long_video_backup_0317.py`에는 별도 실험 축이 있었다.

### (A) Tweedie Caching (`--use_cached`)
- `generate_long_cached()` 사용
- 청크를 순차 처리하면서 overlap의 `x0` 캐시를 재사용

### (B) x_t 캐시 (`--cache_xt`)
- overlap 구간에서 `x_t`까지 재사용해 trajectory 일관성 강화 시도

### (C) velocity blending (`--velocity_blend`)
- 캐시된 velocity와 새 velocity를 블렌딩
- 방법: `linear`, `rbf`, `smoothstep`
- 파라미터: `--blend_method`, `--rbf_gamma`

### (D) soft blend (`--soft_blend`)
- hard replacement 대신 완만한 전이 시도

주의:
- 위 기능은 현재 메인 `generate.py` 경로가 아니라 백업 실험 라인에 남아 있는 코드가 많다.

---

## 6) 현재 상태 요약

- 메인 long video 경로: `generate.py` -> `WanT2VLong.generate_long()`
- 메인 비교 축: `x0_weighted`, `velocity_interp`, `both`, `last_write`
- rollingforcing 배치 실험: 현재 3모드(`x0_weighted`, `velocity_interp`, `both`)
- Tweedie Caching/velocity_blend 계열은 백업 라인 기반 실험 축

---

## 7) 재현 체크리스트

1. 모델 경로 확인: `models/Wan2.1-T2V-1.3B`
2. GPU 확인: `CUDA_VISIBLE_DEVICES=3`
3. 핵심 파라미터 고정:
   - `window_size=81`
   - `multiplier=12` (또는 실험 설정값)
   - `overlap_start=41`
   - `long_steps=25`
4. 비교 축만 변경:
   - `--overlap_mode x0_weighted|velocity_interp|both|last_write`
5. 저장 구조 통일:
   - `prompt_NN/overlap_<mode>/video.mp4`
   - `experiment_log.txt`
