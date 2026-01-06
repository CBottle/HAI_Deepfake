# 📊 HAI Deepfake - 데이터 준비 가이드

**작성일**: 2026-01-06  
**목적**: Google Drive 기반 데이터 수집 및 전처리 완벽 가이드

---

## 🎯 개요

이 가이드는 HAI Deepfake Detection 프로젝트의 학습 데이터를 준비하는 전체 과정을 다룹니다.

### ✅ 장점
- **로컬 저장공간 절약**: 모든 데이터를 Google Drive에 저장
- **Colab 친화적**: GPU 없는 환경에서도 Colab으로 학습 가능
- **단계적 확장**: 소규모로 시작 → 점진적으로 데이터 확대

---

## 📋 준비물

### 1. Kaggle 계정 및 API 토큰
1. https://www.kaggle.com 가입
2. https://www.kaggle.com/settings → "Create New API Token"
3. `kaggle.json` 다운로드

### 2. Google Drive 공간
- 최소 20GB (소규모 테스트)
- 권장 100GB+ (전체 학습)

### 3. Google Colab
- 무료 GPU 사용
- 런타임 유형: GPU (T4 또는 L4)

---

## 🚀 빠른 시작 (Quick Start)

### Step 1: Kaggle API 토큰 업로드
```
Google Drive/
└── MyDrive/
    └── HAI_Deepfake/
        └── kaggle.json  ← 여기에 업로드
```

### Step 2: Colab 노트북 열기
1. Google Colab 접속: https://colab.research.google.com
2. GitHub에서 노트북 열기:
   - URL: `https://github.com/CBottle/HAI_Deepfake`
   - 파일: `notebooks/data_preparation_colab.ipynb`

### Step 3: 순서대로 실행
```python
# 셀을 위에서부터 순서대로 실행
1. 환경 확인
2. Google Drive 마운트
3. 프로젝트 코드 동기화
4. Kaggle API 설정
5. 데이터셋 다운로드  ← 시간 오래 걸림 (10~30분)
6. 비디오 → 이미지 변환
7. 소규모 데이터셋 생성
8. 데이터 검증
```

---

## 📦 추천 데이터셋

### 🥇 소규모 (테스트용) - 추천!

| 데이터셋 | 크기 | 설명 | Kaggle 링크 |
|---------|------|------|-------------|
| **Deepfake and Real Images** | ~1GB | 이미지 형태, 프레임 추출 불필요! | `manjilkarki/deepfake-and-real-images` |

**장점:**
- 이미지 형태라 바로 사용 가능 (비디오 변환 불필요)
- 용량이 작아 빠른 다운로드
- Real/Fake 폴더로 구분되어 있음

**사용 시나리오:**
- 처음 시작할 때
- 모델 아키텍처 실험
- 빠른 프로토타이핑

---

### 🥈 중규모 (균형 잡힌 학습)

| 데이터셋 | 크기 | 설명 | Kaggle 링크 |
|---------|------|------|-------------|
| **Deepfake and Real Images** | ~1GB | 기본 데이터셋 | `manjilkarki/deepfake-and-real-images` |
| **CelebA** | ~1.5GB | Real 얼굴 이미지 200K | `jessicali9530/celeba-dataset` |

**사용 시나리오:**
- 중간 단계 학습
- Real/Fake 밸런스 조정
- 성능 개선 실험

---

### 🥉 대규모 (고성능 목표)

| 데이터셋 | 크기 | 설명 | Kaggle 링크 |
|---------|------|------|-------------|
| **DFDC** | ~470GB | Facebook AI 대규모 딥페이크 | `deepfake-detection-challenge` |
| **CelebA** | ~1.5GB | Real 얼굴 | `jessicali9530/celeba-dataset` |
| **FFHQ** | ~13GB | 고해상도 얼굴 70K | (별도 다운로드) |

**⚠️ 주의:**
- 다운로드 시간: 수 시간
- 저장 공간: 500GB+
- 전처리 시간: 하루 이상

---

## 📂 디렉토리 구조

### Google Drive 구조
```
MyDrive/
└── HAI_Deepfake/
    ├── kaggle.json                 # Kaggle API 토큰
    ├── datasets/                   # Kaggle 원본 데이터
    │   └── deepfake-real-images/   # 다운로드된 데이터셋
    ├── train_data/                 # 학습 데이터 (정리 후)
    │   ├── real/                   # Real 이미지
    │   └── fake/                   # Fake 이미지
    ├── train_data_small/           # 소규모 테스트 데이터
    │   ├── real/                   # Real 1,000장
    │   └── fake/                   # Fake 1,000장
    ├── models/                     # 최종 학습된 모델
    │   └── best_model.pt
    └── checkpoints/                # 학습 중간 체크포인트
        └── 20260106_143022/
            └── checkpoint_epoch_010.pt
```

---

## 🛠️ 상세 사용법

### 1. Kaggle 데이터 다운로드

#### Colab에서 실행:
```python
# 데이터셋 다운로드
!kaggle datasets download -d manjilkarki/deepfake-and-real-images \
    -p /content/drive/MyDrive/HAI_Deepfake/datasets/deepfake-real-images \
    --unzip

# CelebA 추가 (선택)
!kaggle datasets download -d jessicali9530/celeba-dataset \
    -p /content/drive/MyDrive/HAI_Deepfake/datasets/celeba \
    --unzip
```

#### 터미널에서 실행 (고급):
```bash
# Colab 터미널 또는 로컬
kaggle datasets download -d manjilkarki/deepfake-and-real-images -p /path/to/output --unzip
```

---

### 2. 데이터 폴더 구조 정리

**Deepfake and Real Images** 데이터셋은 이미지 형태이므로 **프레임 추출이 불필요**합니다!

다운로드 후 `real/`, `fake/` 폴더 구조로 정리하세요:

```python
# 학습 데이터 폴더 생성
from pathlib import Path

TRAIN_DATA_DIR = Path("/content/drive/MyDrive/HAI_Deepfake/train_data")
(TRAIN_DATA_DIR / "real").mkdir(parents=True, exist_ok=True)
(TRAIN_DATA_DIR / "fake").mkdir(parents=True, exist_ok=True)

# 데이터셋 구조에 맞게 복사 (예시)
# !cp -r datasets/deepfake-real-images/Real/* train_data/real/
# !cp -r datasets/deepfake-real-images/Fake/* train_data/fake/
```

---

### 3. 소규모 데이터셋 생성

#### Colab에서 실행:
```python
# 클래스당 1,000개씩 샘플링
!python scripts/create_small_dataset.py \
    --input "/content/drive/MyDrive/HAI_Deepfake/train_data" \
    --output "/content/drive/MyDrive/HAI_Deepfake/train_data_small" \
    --num-samples 1000 \
    --seed 42
```

#### 비율로 샘플링:
```python
# 전체의 10%
!python scripts/create_small_dataset.py \
    --input train_data \
    --output train_data_small \
    --ratio 0.1 \
    --seed 42
```

---

## 📊 데이터 검증

### Colab에서 샘플 이미지 보기:
```python
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import random

def show_samples(data_dir, num_samples=6):
    data_path = Path(data_dir)
    
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 5))
    
    for idx, label in enumerate(["real", "fake"]):
        label_dir = data_path / label
        images = list(label_dir.glob("*.jpg"))
        samples = random.sample(images, min(num_samples, len(images)))
        
        for i, img_path in enumerate(samples):
            img = Image.open(img_path)
            axes[idx, i].imshow(img)
            axes[idx, i].axis('off')
            axes[idx, i].set_title(f"{label.upper()}")
    
    plt.tight_layout()
    plt.show()

# 실행
show_samples("/content/drive/MyDrive/HAI_Deepfake/train_data_small")
```

---

## ⚠️ 주의사항

### 1. Colab 세션 관리
- **세션 유지**: 브라우저 탭을 닫지 마세요
- **타임아웃**: 90분 idle 시 연결 끊김
- **해결책**: 중간에 체크포인트 저장 (Google Drive에 자동 저장됨)

### 2. Google Drive 용량
- **확인**: https://drive.google.com/settings/storage
- **정리**: 불필요한 파일 삭제
- **업그레이드**: 필요 시 Google One 구독

### 3. 다운로드 시간
| 데이터셋 | 크기 | 예상 시간 |
|---------|------|-----------|
| Deepfake and Real Images | ~1GB | 5~10분 |
| CelebA | 1.5GB | 5~10분 |
| DFDC (전체) | 470GB | 3~6시간 |

### 4. 데이터 라이센스
- **Deepfake and Real Images**: Kaggle 이용 약관 준수
- **CelebA**: 비상업 목적만
- **DFDC**: CC BY-NC-SA 4.0

---

## 🔧 트러블슈팅

### Q1: kaggle.json 파일을 찾을 수 없다고 나와요
```
A: Google Drive의 정확한 경로에 업로드했는지 확인
   위치: /MyDrive/HAI_Deepfake/kaggle.json
```

### Q2: 다운로드가 너무 느려요
```
A: Colab 서버 위치에 따라 다름
   - 런타임 재시작 후 다시 시도
   - 다른 시간대에 시도 (한국 기준 오전 시간 권장)
```

### Q3: 프레임 추출이 오래 걸려요
```
A: 정상입니다
   - 1,000개 비디오 처리: 1~2시간
   - 해결책: --max-videos 10 으로 먼저 테스트
```

### Q4: Google Drive 용량 부족
```
A: 
   1. 소규모 데이터셋만 사용 (train_data_small)
   2. 원본 비디오 삭제 (프레임 추출 후)
   3. Google One 업그레이드 (100GB: 월 $1.99)
```

### Q5: Colab에서 "Runtime disconnected" 에러
```
A:
   1. GPU 리소스 과다 사용 (무료 플랜 제한)
   2. 해결책: Colab Pro 구독 또는 작업 분산
```

---

## 📈 성능 최적화 팁

### 1. 데이터 증강 (학습 시)
```python
# config/config.yaml
data:
  augmentation:
    horizontal_flip: true
    rotation: 10
    color_jitter: true
```

### 2. 균형 잡힌 데이터셋
- Real:Fake = 1:1 비율 유지
- 불균형 시 가중치 조정

### 3. 프레임 품질
- JPEG 품질: 95 (기본)
- 너무 높으면 용량 증가
- 너무 낮으면 성능 저하

---

## 🎯 다음 단계

데이터 준비가 완료되면:

1. **소규모로 학습 시작**
   ```
   notebooks/train_colab.ipynb 실행
   데이터 경로: /content/drive/MyDrive/HAI_Deepfake/train_data_small
   ```

2. **모델 성능 확인**
   - ROC-AUC 점수 확인
   - 과적합 여부 체크

3. **전체 데이터로 확장**
   ```
   데이터 경로: /content/drive/MyDrive/HAI_Deepfake/train_data
   ```

4. **하이퍼파라미터 튜닝**
   - Learning rate
   - Batch size
   - Augmentation

---

## 📚 참고 자료

- **FaceForensics++ 논문**: https://arxiv.org/abs/1901.08971
- **DFDC 대회**: https://ai.facebook.com/datasets/dfdc/
- **CelebA 데이터셋**: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- **Kaggle API 문서**: https://github.com/Kaggle/kaggle-api

---

**작성자**: OpenCode AI  
**마지막 업데이트**: 2026-01-06  
**버전**: 1.0
