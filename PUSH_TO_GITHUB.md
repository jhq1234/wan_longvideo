# GitHub에 푸시하는 방법

레포지토리 준비가 완료되었습니다. 다음 단계를 따라 GitHub에 업로드하세요:

## 방법 1: GitHub 웹사이트에서 레포지토리 생성 후 푸시

1. https://github.com/new 에서 레포지토리 생성:
   - Repository name: `wan_longvideo`
   - Description: `Long video generation experiments using Wan2.1-T2V models`
   - Public 또는 Private 선택
   - **Initialize this repository with a README 체크 해제** (이미 README가 있음)

2. 다음 명령어 실행:
```bash
cd /home/bispl_02/jangho/wan_longvideo
git push -u origin main
```

## 방법 2: GitHub CLI 사용 (권장)

GitHub CLI가 설치되어 있다면:
```bash
cd /home/bispl_02/jangho/wan_longvideo
gh repo create jhq1234/wan_longvideo --public --source=. --remote=origin --push
```

## 방법 3: Personal Access Token 사용

GitHub Personal Access Token이 있다면:
```bash
cd /home/bispl_02/jangho/wan_longvideo
git push -u https://YOUR_TOKEN@github.com/jhq1234/wan_longvideo.git main
```

## 포함된 파일들

- `long_video.py`: Long video generation 구현 코드
- `video_synctweedies_protocol.md`: SyncTweedies 프로토콜 문서
- `run_long_video_experiments.sh`: 실험 실행 스크립트
- `prompts_30.txt`: 사용된 프롬프트 모음
- `output_*/`: 생성된 비디오 결과들 (약 50개 파일, 총 930MB)
