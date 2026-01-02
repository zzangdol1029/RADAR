# Git에서 Ignore된 파일 제거 및 푸시 가이드

## 🚀 빠른 실행

```bash
# 자동 스크립트 사용 (권장)
./remove_ignored_files.sh
```

## 📝 수동 실행 방법

### 1단계: Git에 추가된 JSON 파일 확인

```bash
# JSON 파일 확인
git ls-files | grep "\.json$"

# 또는 전처리 파일만
git ls-files | grep "preprocessed_logs.*\.json"
```

### 2단계: Git에서 제거 (파일은 유지)

```bash
# 모든 JSON 파일 제거
git ls-files | grep "\.json$" | xargs git rm --cached

# 또는 전처리 파일만
git ls-files | grep "preprocessed_logs.*\.json" | xargs git rm --cached

# 또는 특정 파일만
git rm --cached preprocessing/output/preprocessed_logs_2025-02-25.json
```

### 3단계: .gitignore 확인

```bash
# .gitignore에 JSON 규칙이 있는지 확인
cat .gitignore | grep -E "\.json|preprocessed"

# 없으면 추가
echo "*.json" >> .gitignore
echo "preprocessing/output/preprocessed_logs_*.json" >> .gitignore
```

### 4단계: 변경사항 커밋

```bash
# .gitignore도 함께 커밋
git add .gitignore
git commit -m "JSON 파일을 Git에서 제거 (용량 문제로 .gitignore에 추가)"
```

### 5단계: 푸시

```bash
# 푸시
git push -u origin main

# 또는
git push
```

## 🔍 확인 명령어

```bash
# Git 상태 확인
git status

# 제거된 파일 확인
git status --short

# .gitignore가 제대로 작동하는지 확인
git check-ignore preprocessing/output/preprocessed_logs_*.json
```

## ⚠️ 주의사항

1. **파일은 유지됩니다**: `git rm --cached`는 Git에서만 제거하고 로컬 파일은 그대로 유지합니다.

2. **이미 푸시된 파일**: 원격 저장소에 이미 푸시된 파일은 이 커밋으로 제거됩니다.

3. **협업 중인 경우**: 다른 사람이 pull 받기 전에 푸시하는 것이 좋습니다.

## 💡 전체 프로세스 (한 번에)

```bash
# 1. JSON 파일 제거
git ls-files | grep "\.json$" | xargs git rm --cached

# 2. .gitignore 확인/추가
echo "*.json" >> .gitignore
echo "preprocessing/output/preprocessed_logs_*.json" >> .gitignore

# 3. 커밋
git add .gitignore
git commit -m "JSON 파일을 Git에서 제거"

# 4. 푸시
git push -u origin main
```

## 🛠️ 문제 해결

### 파일이 여전히 추적되는 경우

```bash
# 강제로 제거
git rm --cached -r preprocessing/output/

# 또는 특정 파일
git rm --cached preprocessing/output/preprocessed_logs_*.json
```

### .gitignore가 작동하지 않는 경우

```bash
# Git 캐시 정리
git rm -r --cached .
git add .
git commit -m "Git 캐시 정리 및 .gitignore 적용"
```

### 이미 푸시된 파일 제거

```bash
# 원격에서도 제거 (주의: 협업 중이면 팀원과 상의)
git rm --cached preprocessing/output/preprocessed_logs_*.json
git commit -m "JSON 파일 제거"
git push -u origin main
```

## 📋 체크리스트

- [ ] Git에 추가된 JSON 파일 확인
- [ ] `git rm --cached`로 Git에서 제거
- [ ] .gitignore에 JSON 규칙 확인/추가
- [ ] 변경사항 커밋
- [ ] 원격 저장소에 푸시
- [ ] 파일이 로컬에 유지되는지 확인







