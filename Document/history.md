# train_colab.ipynb 개발 히스토리

**작성일:** 2025-01-05

---

## 1. 문제 정의

### 상황
딥페이크 탐지 모델을 학습시키려면 **GPU가 필수**입니다. 하지만 대부분의 개인 컴퓨터에는 고성능 GPU가 없거나, 있어도 VRAM이 부족합니다.

### Google Colab의 등장
Google Colab은 **무료 GPU**를 제공하는 클라우드 환경입니다. 하지만 Colab에는 치명적인 단점들이 있습니다:

| 문제 | 설명 |
|------|------|
| **휘발성 환경** | Colab은 세션이 끊기면 모든 파일이 삭제됨 |
| **코드 동기화 어려움** | 로컬에서 수정한 코드를 Colab에 반영하려면 매번 파일을 업로드해야 함 |
| **환경 초기화** | 매번 새 세션마다 라이브러리를 다시 설치해야 함 |
| **대용량 데이터** | 딥페이크 데이터셋은 수십 GB → 업로드에 시간이 오래 걸림 |
| **학습 결과 손실** | 세션 종료 시 학습된 모델(.pt 파일)도 함께 삭제됨 |

---

## 2. 해결 방법

### train_colab.ipynb의 4가지 핵심 임무

```
┌─────────────────────────────────────────────────────────────┐
│                    train_colab.ipynb                        │
├─────────────────────────────────────────────────────────────┤
│  1. 코드 동기화 (Git Sync)                                  │
│     └─ GitHub에서 최신 코드를 한 번에 가져옴                │
│                                                             │
│  2. 자동 환경 세팅 (Environment Setup)                      │
│     └─ requirements.txt로 라이브러리 자동 설치              │
│                                                             │
│  3. 데이터 고속도로 (Data Loading)                          │
│     └─ Kaggle API로 서버 간 직접 다운로드 (초고속)          │
│                                                             │
│  4. 금고 보관 (Model Persistence)                           │
│     └─ Google Drive에 모델 자동 백업                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 구현 내용

### 3.1 코드 동기화 (Git Sync)

**문제:** 로컬에서 코드를 수정할 때마다 Colab에 파일을 일일이 업로드해야 함

**해결:**
```python
REPO_URL = "https://github.com/CBottle/HAI_Deepfake.git"

if os.path.exists(PROJECT_DIR):
    # 이미 클론되어 있으면 최신 코드로 업데이트
    !git fetch --all
    !git reset --hard origin/main
else:
    # 처음이면 클론
    !git clone {REPO_URL}
```

**사용법:**
1. 로컬에서 코드 수정 후 `git push`
2. Colab에서 해당 셀 하나만 실행
3. 최신 코드가 자동으로 동기화됨

---

### 3.2 자동 환경 세팅 (Environment Setup)

**문제:** Colab은 새 세션마다 빈 컴퓨터 상태로 시작 → 매번 pip install 필요

**해결:**
```python
req_paths = ['requirements.txt', 'env/requirements.txt']

for path in req_paths:
    if os.path.exists(path):
        !pip install -q -r {path}
        break
```

**효과:** 셀 하나로 모든 의존성 자동 설치

---

### 3.3 데이터 고속도로 (Data Loading)

**문제:** 딥페이크 데이터셋은 수십 GB → 내 컴퓨터를 거치면 업로드에 몇 시간 소요

**해결:**
```
[기존 방식]
Kaggle 서버 → 내 컴퓨터 → Colab 서버  (느림)

[Kaggle API 방식]
Kaggle 서버 ────────────→ Colab 서버  (빠름)
```

```python
# Kaggle API 키 설정
!cp /content/drive/MyDrive/Kaggle/kaggle.json /root/.kaggle/

# 서버 간 직접 다운로드
!kaggle datasets download -d {DATASET_NAME} -p ./train_data --unzip
```

**사전 준비:**
1. Kaggle 계정에서 `kaggle.json` 다운로드
2. Google Drive의 `/MyDrive/Kaggle/kaggle.json`에 업로드

---

### 3.4 금고 보관 (Model Persistence)

**문제:** Colab 세션 종료 시 학습된 모델이 모두 삭제됨

**해결:**
```python
from google.colab import drive
drive.mount('/content/drive')

# 타임스탬프로 백업 폴더 생성
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
BACKUP_PATH = f"/content/drive/MyDrive/HAI_Deepfake/checkpoints/{timestamp}"

# 체크포인트 복사
shutil.copy2(src, dst)
```

**결과:**
```
Google Drive/
└── HAI_Deepfake/
    └── checkpoints/
        ├── 20250105_143022/
        │   └── checkpoint_epoch_001.pt
        └── 20250105_180512/
            └── best_model.pt
```

---

## 4. 파일 구조

```
HAI_Deepfake/
├── train_colab.ipynb    ← Colab 전용 학습 파이프라인 (NEW)
├── train.py             ← 실제 학습 로직
├── config/
│   └── config.yaml      ← 학습 설정
├── env/
│   └── requirements.txt ← 의존성 목록
└── src/
    ├── models.py        ← 모델 정의
    ├── dataset.py       ← 데이터셋 처리
    └── utils.py         ← 유틸리티 함수
```

---

## 5. 사용 가이드

### 최초 실행
1. Colab에서 `train_colab.ipynb` 열기
2. **런타임 > 런타임 유형 변경 > GPU** 선택
3. **런타임 > 모두 실행** (Ctrl+F9)
4. Google Drive 마운트 승인

### 코드 업데이트 시
1. 로컬에서 코드 수정
2. `git add . && git commit -m "메시지" && git push`
3. Colab에서 **섹션 2 (코드 동기화)** 셀만 재실행

### 학습 재개 시
- Google Drive에 백업된 체크포인트로 학습 재개 가능
- `train.py --resume` 옵션 사용

---

## 6. 요약

| Before | After |
|--------|-------|
| 코드 수정마다 파일 업로드 | `git push` → 셀 실행 |
| 매번 pip install 수동 실행 | 자동 설치 |
| 데이터 업로드에 몇 시간 | Kaggle API로 직접 다운로드 |
| 세션 종료 시 모델 손실 | Google Drive에 자동 백업 |

**train_colab.ipynb는 Colab의 단점을 보완하여 효율적인 딥러닝 학습 환경을 제공합니다.**
