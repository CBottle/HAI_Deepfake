# AGENTS.md - Coding Agent Guidelines

이 문서는 HAI Deepfake Detection 프로젝트에서 작업하는 AI 코딩 에이전트를 위한 가이드입니다.

---

## 1. 프로젝트 개요

**목적**: 딥페이크 이미지/비디오 탐지 (HAI 대회 제출용)  
**주요 기술**: PyTorch 2.5.0, Vision Transformer (ViT), Hugging Face Transformers  
**언어**: Python 3.10  
**제약사항**:
- 단일 모델만 허용 (앙상블 금지)
- 이미지 레벨 입력 (멀티프레임 시간 모델 불가)
- 추론 시간 제한: 60분
- VRAM: 48GB 요구
- 오프라인 동작 (API 호출 불가)

---

## 2. Build / Lint / Test Commands

### 환경 설정
```bash
# Conda 환경 생성
conda env create -f env/environment.yml
conda activate deepfake

# 또는 pip로 설치
pip install -r env/requirements.txt

# 더미 데이터 생성 (로컬 테스트용)
python create_dummy_data.py
```

### 학습 실행
```bash
# 전체 학습
python train.py --config config/config.yaml

# 디버그 모드 (빠른 테스트용)
python train.py --debug --epochs 1 --batch_size 2 --device cpu

# 체크포인트에서 재개
python train.py --resume output/checkpoint_epoch_001.pt
```

### 추론 실행
```bash
# 기본 추론
python inference.py --config config/config.yaml --test_dir test_data --output submissions/submission.csv

# 특정 모델로 추론
python inference.py --checkpoint model/best_model.pt --test_dir test_data
```

### Docker 실행
```bash
# 이미지 빌드
docker build -t hai-deepfake:latest -f env/Dockerfile .

# 추론 실행
docker run --gpus all -v $(pwd)/test_data:/workspace/test_data hai-deepfake:latest python inference.py --test_dir test_data
```

### 테스트
**주의**: 현재 공식 테스트 프레임워크(pytest, unittest)는 설정되어 있지 않습니다.
- Jupyter 노트북 (`notebooks/`) 을 통한 대화형 테스트
- `--debug` 플래그로 빠른 검증

---

## 3. 코드 스타일 가이드라인

### 3.1 Import 순서
1. **표준 라이브러리** (알파벳 순)
2. **서드파티 라이브러리** (torch, transformers, sklearn 등)
3. **로컬 모듈** (`from src.module import ...`)

**예시**:
```python
import argparse
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from transformers import ViTImageProcessor
from tqdm import tqdm

from src.models import DeepfakeDetector
from src.dataset import DeepfakeDataset
from src.utils import set_seed, load_config
```

### 3.2 네이밍 컨벤션
- **파일명**: `snake_case` (예: `dataset.py`, `train.py`)
- **클래스명**: `PascalCase` (예: `DeepfakeDetector`, `InferenceDataset`)
- **함수/변수명**: `snake_case` (예: `train_epoch`, `pixel_values`)
- **상수**: `UPPER_SNAKE_CASE` (예: `IMAGE_EXTS`, `VIDEO_EXTS`)
- **Private 메서드**: `_single_underscore` (예: `_collect_samples`, `_read_frames`)

### 3.3 타입 힌트
모든 함수에 타입 어노테이션을 필수로 사용하세요.

```python
from typing import List, Optional, Tuple, Dict

def load_config(config_path: str) -> dict:
    """설정 파일 로드"""
    ...

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> float:
    """한 에포크 학습"""
    ...
```

### 3.4 Docstring 스타일
**Google 스타일**을 사용하며, **한국어**로 작성합니다.

```python
def validate(model, dataloader, criterion, device):
    """
    검증 수행
    
    Args:
        model: 검증할 모델
        dataloader: 검증 데이터 로더
        criterion: 손실 함수
        device: 디바이스 (cuda/cpu)
    
    Returns:
        평균 손실, ROC-AUC 점수
    """
    ...
```

### 3.5 포매팅
- **들여쓰기**: 4 스페이스
- **라인 길이**: 합리적 범위 내 (엄격한 제한 없음)
- **클래스/함수 사이**: 2줄 띄우기
- **메서드 사이**: 1줄 띄우기

### 3.6 에러 처리
- 데이터 로딩 실패 시 빈 텐서 반환 (크래시 방지)
- Try-except 블록 사용 시 구체적인 예외 타입 지정 권장

```python
def _read_frames(self, file_path: Path) -> List[np.ndarray]:
    """이미지 또는 비디오에서 프레임 추출"""
    try:
        img = Image.open(file_path).convert("RGB")
        return [np.array(img)]
    except Exception:
        return []  # 실패 시 빈 리스트
```

---

## 4. 프로젝트 구조

```
HAI_Deepfake/
├── src/                    # 소스 모듈 (핵심 코드)
│   ├── models.py          # 모델 아키텍처
│   ├── dataset.py         # 데이터셋 & 전처리
│   └── utils.py           # 유틸리티 함수
├── config/
│   └── config.yaml        # 학습/추론 설정
├── env/                   # 환경 설정
│   ├── Dockerfile
│   ├── requirements.txt
│   └── environment.yml
├── notebooks/             # Jupyter 노트북 (개발용)
├── Document/              # 문서
├── train_data/            # 학습 데이터 (real/, fake/)
├── test_data/             # 테스트 데이터
├── output/                # 체크포인트 저장
├── submissions/           # CSV 제출 파일
├── train.py               # 학습 엔트리 포인트
├── inference.py           # 추론 엔트리 포인트
└── create_dummy_data.py   # 더미 데이터 생성
```

---

## 5. 중요 개발 패턴

### 5.1 Path 처리
**항상 `pathlib.Path` 사용** (문자열 경로 지양)

```python
from pathlib import Path

data_dir = Path("train_data")
for file_path in data_dir.glob("*.jpg"):
    print(file_path.name)
```

### 5.2 Device 처리
`src.utils.get_device()` 함수 사용

```python
from src.utils import get_device

device = get_device()  # 자동으로 CUDA/CPU 선택
model = model.to(device)
```

### 5.3 재현성 확보
모든 스크립트에서 `set_seed()` 호출

```python
from src.utils import set_seed

set_seed(42)  # 고정 시드
```

### 5.4 설정 관리
YAML 파일 기반 설정 사용

```python
from src.utils import load_config

config = load_config("config/config.yaml")
lr = config['training']['learning_rate']
```

### 5.5 체크포인트 저장/로드
```python
from src.utils import save_checkpoint, load_checkpoint

# 저장
save_checkpoint(model, optimizer, epoch, val_auc=0.95, checkpoint_dir="output")

# 로드
checkpoint = load_checkpoint("output/checkpoint_epoch_010.pt", model, optimizer, device)
start_epoch = checkpoint['epoch'] + 1
```

### 5.6 Progress Bar
tqdm 사용

```python
from tqdm import tqdm

for batch in tqdm(dataloader, desc='Training'):
    ...
```

---

## 6. 데이터 처리 규칙

### 6.1 지원 파일 형식
- **이미지**: `.jpg`, `.jpeg`, `.png`, `.jfif`
- **비디오**: `.mp4`, `.mov`

### 6.2 데이터 디렉토리 구조 (학습)
```
train_data/
├── real/
│   ├── image1.jpg
│   └── video1.mp4
└── fake/
    ├── image2.jpg
    └── video2.mp4
```

### 6.3 레이블 매핑
```python
class_to_idx = {"real": 0, "fake": 1}
```

### 6.4 비디오 프레임 샘플링
- 균등 간격으로 `num_frames` 개 추출 (기본: 10)
- 단일 이미지 모델이므로 실제로는 첫 프레임만 사용

---

## 7. 체크리스트 (코드 작성 시 확인 사항)

### 코드 작성 전
- [ ] 타입 힌트를 모든 함수에 추가했는가?
- [ ] Docstring을 한국어로 작성했는가? (Args/Returns 포함)
- [ ] Import 순서가 올바른가? (표준 라이브러리 → 서드파티 → 로컬)

### 모델 관련
- [ ] `DeepfakeDetector` 클래스 사용
- [ ] 사전학습 모델: `google/vit-base-patch16-224` 또는 `prithivMLmods/Deep-Fake-Detector-v2-Model`
- [ ] 출력: 2개 클래스 (Real/Fake)

### 학습 관련
- [ ] `set_seed()`로 재현성 확보
- [ ] Mixed Precision (AMP) 사용 (GPU 학습 시)
- [ ] 체크포인트 정기 저장 (config에서 설정)
- [ ] tqdm으로 진행 상황 표시

### 데이터 관련
- [ ] `DeepfakeDataset` (학습) 또는 `InferenceDataset` (추론) 사용
- [ ] 에러 발생 시 빈 텐서 반환 (크래시 방지)
- [ ] `ViTImageProcessor`로 전처리

### 추론 관련
- [ ] 출력: CSV 파일 (`filename,label` 형식)
- [ ] Label: 0 (Real) / 1 (Fake)
- [ ] Softmax 확률에서 `argmax` 또는 임계값 사용

### 제출 전 (대회)
- [ ] 단일 모델만 사용 (앙상블 금지)
- [ ] 추론 시간 60분 이내
- [ ] 모든 코드가 `.py` 파일 (노트북 제외)
- [ ] 오프라인 동작 확인

---

## 8. Git 브랜치 전략

- `main`: 안정 버전
- `dev`: 개발 브랜치
- `feature/*`: 기능 개발
- `fix/*`: 버그 수정

---

## 9. 참고 문서

- **프로젝트 README**: `README.md`
- **환경 설정 가이드**: `Document/SETUP.md`
- **대회 규칙**: `Document/Rule.md`
- **개발 히스토리**: `Document/history.md`
- **베이스라인 노트북**: `Document/baseline.ipynb`

---

## 10. 자주 사용하는 설정값 (config.yaml)

```yaml
model:
  name: "google/vit-base-patch16-224"
  num_classes: 2

training:
  epochs: 50
  batch_size: 32
  learning_rate: 1e-4
  weight_decay: 1e-4
  mixed_precision: true

data:
  num_frames: 10
  image_size: 224
```

---

**작성일**: 2026-01-06  
**버전**: 1.0  
**유지보수**: 프로젝트 변경 사항 발생 시 이 문서를 업데이트하세요.
