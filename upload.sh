#!/bin/bash
# GitHub에 레포지토리 생성 및 푸시 스크립트

echo "GitHub 레포지토리 생성 및 푸시 중..."

# GitHub CLI로 레포지토리 생성 및 푸시
gh repo create jhq1234/wan_longvideo \
  --public \
  --description "Long video generation experiments using Wan2.1-T2V models" \
  --source=. \
  --remote=origin \
  --push

if [ $? -eq 0 ]; then
    echo "✅ 성공적으로 업로드되었습니다!"
    echo "🔗 https://github.com/jhq1234/wan_longvideo"
else
    echo "❌ 업로드 실패. GitHub CLI 인증이 필요할 수 있습니다."
    echo "다음 명령어로 인증하세요: gh auth login"
fi
