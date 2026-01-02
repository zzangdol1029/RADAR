#!/bin/bash
# Git에서 ignore된 파일 제거 및 푸시 스크립트

set -e

echo "=========================================="
echo "Git에서 ignore된 파일 제거 및 푸시"
echo "=========================================="
echo ""

# 1. 현재 상태 확인
echo "1. 현재 Git 상태 확인 중..."
git status --short | head -20
echo ""

# 2. 이미 추가된 JSON 파일 확인
echo "2. Git에 추가된 JSON 파일 확인 중..."
json_files=$(git ls-files | grep -E "\.json$|preprocessed_logs.*\.json$" || true)

if [ -z "$json_files" ]; then
    echo "✅ Git에 추가된 JSON 파일이 없습니다."
    echo ""
    echo "현재 상태 확인:"
    git status
    exit 0
fi

echo "다음 JSON 파일들이 Git에 추가되어 있습니다:"
echo "$json_files" | while read file; do
    if [ -n "$file" ]; then
        size=$(git ls-files -s "$file" | awk '{print $4}' | xargs git cat-file -s 2>/dev/null || echo "unknown")
        echo "  - $file (Git 크기: $size bytes)"
    fi
done
echo ""

# 3. 확인
read -p "이 파일들을 Git에서 제거하시겠습니까? (파일 자체는 유지됩니다) (y/n): " confirm
if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "취소되었습니다."
    exit 0
fi

# 4. JSON 파일들을 Git에서 제거 (파일은 유지)
echo ""
echo "3. JSON 파일들을 Git에서 제거 중..."
echo "$json_files" | while read file; do
    if [ -n "$file" ]; then
        echo "  제거: $file"
        git rm --cached "$file" 2>/dev/null || true
    fi
done

# 모든 JSON 파일 제거 (더 안전한 방법)
git ls-files | grep -E "\.json$" | xargs -r git rm --cached 2>/dev/null || true
git ls-files | grep "preprocessed_logs.*\.json" | xargs -r git rm --cached 2>/dev/null || true

echo "✅ Git에서 제거 완료"
echo ""

# 5. .gitignore 확인
echo "4. .gitignore 확인 중..."
if grep -q "\.json" .gitignore || grep -q "preprocessed_logs" .gitignore; then
    echo "✅ .gitignore에 JSON 파일 규칙이 있습니다."
else
    echo "⚠️  .gitignore에 JSON 파일 규칙이 없습니다."
    read -p ".gitignore에 추가하시겠습니까? (y/n): " add_ignore
    if [ "$add_ignore" = "y" ] || [ "$add_ignore" = "Y" ]; then
        echo "" >> .gitignore
        echo "# JSON 파일" >> .gitignore
        echo "*.json" >> .gitignore
        echo "preprocessing/output/preprocessed_logs_*.json" >> .gitignore
        echo "✅ .gitignore에 추가 완료"
    fi
fi
echo ""

# 6. 변경사항 확인
echo "5. 변경사항 확인:"
git status --short
echo ""

# 7. 커밋
read -p "변경사항을 커밋하시겠습니까? (y/n): " commit_confirm
if [ "$commit_confirm" = "y" ] || [ "$commit_confirm" = "Y" ]; then
    echo ""
    echo "6. 커밋 중..."
    git add .gitignore
    git commit -m "JSON 파일을 Git에서 제거 (용량 문제로 .gitignore에 추가)"
    echo "✅ 커밋 완료"
    echo ""
    
    # 8. 푸시
    read -p "원격 저장소에 푸시하시겠습니까? (y/n): " push_confirm
    if [ "$push_confirm" = "y" ] || [ "$push_confirm" = "Y" ]; then
        echo ""
        echo "7. 푸시 중..."
        git push -u origin main || git push
        echo "✅ 푸시 완료"
    else
        echo "푸시를 건너뜁니다."
        echo "나중에 다음 명령어로 푸시하세요:"
        echo "  git push -u origin main"
    fi
else
    echo "커밋을 건너뜁니다."
    echo ""
    echo "다음 명령어로 수동으로 커밋 및 푸시하세요:"
    echo "  git add .gitignore"
    echo "  git commit -m 'JSON 파일을 Git에서 제거'"
    echo "  git push -u origin main"
fi

echo ""
echo "=========================================="
echo "완료!"
echo "=========================================="
echo ""
echo "참고:"
echo "- JSON 파일들은 로컬에 그대로 유지됩니다"
echo "- 이제 Git에서 무시되므로 푸시되지 않습니다"
echo "- .gitignore에 JSON 파일 규칙이 추가되었습니다"







