#!/bin/bash
# Ignore된 파일은 그대로 두고 다른 변경사항만 푸시하는 스크립트

set -e

echo "=========================================="
echo "Ignore된 파일 유지하고 푸시"
echo "=========================================="
echo ""

# 1. .gitignore 확인
echo "1. .gitignore 확인 중..."
if grep -q "\.json" .gitignore || grep -q "preprocessed_logs" .gitignore; then
    echo "✅ .gitignore에 JSON 파일 규칙이 있습니다."
    echo ""
    echo "현재 .gitignore 규칙:"
    grep -E "\.json|preprocessed" .gitignore | sed 's/^/  /'
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

# 2. Git 상태 확인
echo "2. Git 상태 확인 중..."
echo "----------------------------------------"
git status --short
echo ""

# 3. Ignore된 파일 확인
echo "3. Ignore된 파일 확인 중..."
ignored_files=$(git status --ignored --short | grep "!!" | awk '{print $2}' | head -10 || true)
if [ -n "$ignored_files" ]; then
    echo "다음 파일들이 무시되고 있습니다:"
    echo "$ignored_files" | while read file; do
        if [ -n "$file" ]; then
            echo "  - $file"
        fi
    done
    echo ""
    echo "✅ 이 파일들은 푸시되지 않습니다."
else
    echo "✅ 무시된 파일이 없습니다."
fi
echo ""

# 4. 추적 중인 파일 확인
echo "4. Git에 추적 중인 파일 확인 중..."
tracked_json=$(git ls-files | grep -E "\.json$" | head -5 || true)
if [ -n "$tracked_json" ]; then
    echo "⚠️  다음 JSON 파일들이 Git에 추적 중입니다:"
    echo "$tracked_json" | while read file; do
        if [ -n "$file" ]; then
            echo "  - $file"
        fi
    done
    echo ""
    echo "이 파일들은 .gitignore에 추가되어도 계속 추적됩니다."
    echo "제거하려면: git rm --cached <파일명>"
    echo ""
    read -p "이 파일들을 Git에서 제거하시겠습니까? (y/n): " remove_tracked
    if [ "$remove_tracked" = "y" ] || [ "$remove_tracked" = "Y" ]; then
        echo "$tracked_json" | while read file; do
            if [ -n "$file" ]; then
                git rm --cached "$file" 2>/dev/null || true
            fi
        done
        echo "✅ Git에서 제거 완료"
    fi
else
    echo "✅ 추적 중인 JSON 파일이 없습니다."
fi
echo ""

# 5. 커밋할 변경사항 확인
echo "5. 커밋할 변경사항 확인 중..."
changes=$(git status --short | grep -v "^!!" | grep -v "\.json$" || true)
if [ -z "$changes" ]; then
    echo "⚠️  커밋할 변경사항이 없습니다."
    echo ""
    read -p ".gitignore 변경사항만 커밋하시겠습니까? (y/n): " commit_ignore
    if [ "$commit_ignore" = "y" ] || [ "$commit_ignore" = "Y" ]; then
        if git diff --quiet .gitignore; then
            echo "✅ .gitignore에 변경사항이 없습니다."
        else
            git add .gitignore
            git commit -m "JSON 파일을 .gitignore에 추가"
            echo "✅ .gitignore 커밋 완료"
        fi
    fi
else
    echo "다음 변경사항이 있습니다:"
    echo "$changes" | sed 's/^/  /'
    echo ""
    read -p "이 변경사항들을 커밋하시겠습니까? (y/n): " commit_confirm
    if [ "$commit_confirm" = "y" ] || [ "$commit_confirm" = "Y" ]; then
        # .gitignore도 함께 추가
        if ! git diff --quiet .gitignore; then
            git add .gitignore
        fi
        
        # 다른 변경사항 추가 (JSON 제외)
        git add -A
        git reset HEAD -- "*.json" 2>/dev/null || true
        git reset HEAD -- "preprocessing/output/preprocessed_logs_*.json" 2>/dev/null || true
        
        # 커밋
        git commit -m "변경사항 커밋 (JSON 파일 제외)"
        echo "✅ 커밋 완료"
    fi
fi
echo ""

# 6. 푸시
echo "6. 원격 저장소에 푸시 중..."
read -p "푸시하시겠습니까? (y/n): " push_confirm
if [ "$push_confirm" = "y" ] || [ "$push_confirm" = "Y" ]; then
    git push -u origin main || git push
    echo "✅ 푸시 완료"
    echo ""
    echo "참고: JSON 파일들은 .gitignore에 의해 무시되어 푸시되지 않았습니다."
else
    echo "푸시를 건너뜁니다."
    echo "나중에 다음 명령어로 푸시하세요:"
    echo "  git push -u origin main"
fi

echo ""
echo "=========================================="
echo "완료!"
echo "=========================================="
echo ""
echo "요약:"
echo "- JSON 파일들은 .gitignore에 의해 무시됩니다"
echo "- JSON 파일들은 푸시되지 않습니다"
echo "- 다른 변경사항만 푸시되었습니다"







