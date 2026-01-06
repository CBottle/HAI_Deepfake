# 🔑 Kaggle API 설정 가이드

## 받은 키값으로 kaggle.json 만들기

---

## 📋 받은 정보

Kaggle Settings에서 받은 정보:
```
Username: your_username
Key: xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

---

## 🛠️ 설정 방법

### 방법 1: 직접 파일 생성 (Windows - 로컬)

#### Step 1: 메모장으로 파일 생성
```
1. 메모장 열기 (notepad)
2. 아래 내용 복사해서 붙여넣기:

{
  "username": "받은_username_여기에",
  "key": "받은_key_여기에"
}

3. 다른 이름으로 저장
   - 파일명: kaggle.json (따옴표 없이!)
   - 파일 형식: 모든 파일 (*.*)
   - 저장 위치: C:\Users\Jun\OneDrive\Desktop\STUDY\HAI_Deepfake\
```

**예시:**
```json
{
  "username": "cbottle",
  "key": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"
}
```

---

### 방법 2: Colab에서 직접 입력 (추천!)

Colab 노트북에서 아래 코드 실행:

```python
import json
from pathlib import Path

# 여기에 받은 값 입력
KAGGLE_USERNAME = "받은_username"  # ← 여기 수정
KAGGLE_KEY = "받은_key"           # ← 여기 수정

# kaggle.json 생성
kaggle_config = {
    "username": KAGGLE_USERNAME,
    "key": KAGGLE_KEY
}

# Google Drive에 저장
drive_path = Path("/content/drive/MyDrive/HAI_Deepfake")
drive_path.mkdir(parents=True, exist_ok=True)

kaggle_json_path = drive_path / "kaggle.json"
with open(kaggle_json_path, 'w') as f:
    json.dump(kaggle_config, f, indent=2)

print(f"✅ kaggle.json 생성 완료!")
print(f"📁 위치: {kaggle_json_path}")
```

---

### 방법 3: PowerShell로 생성 (로컬 - 고급)

PowerShell에서 실행:

```powershell
# 변수 설정 (값 수정 필요)
$username = "받은_username"
$key = "받은_key"

# JSON 생성
$kaggleConfig = @{
    username = $username
    key = $key
} | ConvertTo-Json

# 파일 저장
$kaggleConfig | Out-File -FilePath "kaggle.json" -Encoding utf8

Write-Host "✅ kaggle.json 생성 완료!"
```

---

## 📤 Google Drive에 업로드

### Windows에서:
```
1. Google Drive 웹사이트 접속
   https://drive.google.com

2. 폴더 생성
   - 내 드라이브 > 새로 만들기 > 폴더
   - 이름: HAI_Deepfake

3. kaggle.json 업로드
   - HAI_Deepfake 폴더로 이동
   - 파일 업로드 클릭
   - kaggle.json 선택

최종 위치: /MyDrive/HAI_Deepfake/kaggle.json
```

---

## ✅ 확인 방법

### Colab에서 확인:

```python
from pathlib import Path
import json

# 파일 존재 확인
kaggle_json = Path("/content/drive/MyDrive/HAI_Deepfake/kaggle.json")

if kaggle_json.exists():
    print("✅ kaggle.json 파일 발견!")
    
    # 내용 확인 (키는 숨김)
    with open(kaggle_json) as f:
        config = json.load(f)
    
    print(f"Username: {config['username']}")
    print(f"Key: {config['key'][:10]}...")  # 앞 10자만
else:
    print("❌ 파일을 찾을 수 없습니다.")
    print(f"위치: {kaggle_json}")
```

---

## 🔒 보안 주의사항

### ⚠️ 중요!
```
1. kaggle.json은 비밀 정보!
   - GitHub에 절대 업로드 금지
   - 공개 저장소에 공유 금지

2. .gitignore에 추가됨 확인
   - 프로젝트 .gitignore에 kaggle.json 포함됨

3. 유출 시 조치
   - Kaggle Settings에서 토큰 재발급
   - 기존 토큰 무효화
```

---

## 🐛 트러블슈팅

### Q1: JSON 형식 오류
```
A: 중괄호 { } 확인
   쉼표 , 확인 (마지막 항목 뒤에는 쉼표 없음)
   따옴표 " " 확인 (작은따옴표 ' 아님)
```

### Q2: 파일 확장자가 .json.txt로 저장됨
```
A: Windows 메모장 사용 시 주의
   - 저장 시 "모든 파일 (*.*)" 선택
   - 파일명에 확장자 직접 입력: kaggle.json
   - 또는 VSCode 같은 편집기 사용
```

### Q3: Colab에서 파일을 찾을 수 없음
```
A: Google Drive 경로 확인
   1. Google Drive 마운트 확인
   2. 폴더명 대소문자 확인 (HAI_Deepfake)
   3. 파일명 확인 (kaggle.json, 공백 없음)
```

---

## 📝 빠른 체크리스트

- [ ] Kaggle Username과 Key 받음
- [ ] kaggle.json 파일 생성
- [ ] JSON 형식 올바른지 확인
- [ ] Google Drive HAI_Deepfake 폴더에 업로드
- [ ] Colab에서 파일 존재 확인
- [ ] Kaggle API 테스트 (데이터셋 목록 조회)

---

## 🚀 다음 단계

kaggle.json 설정 완료 후:

```
✅ data_preparation_colab.ipynb 실행
✅ Step 4에서 자동으로 Kaggle API 설정
✅ Step 5에서 데이터 다운로드 시작!
```

---

**작성일**: 2026-01-06  
**도움말**: 문제 생기면 언제든 물어보세요!
