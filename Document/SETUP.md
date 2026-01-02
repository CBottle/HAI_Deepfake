# HAI Deepfake - 개발환경 설정 가이드

> 딥페이크 탐지 AI 모델 개발을 위한 개발환경 설정 문서

## 📋 목차

1. [환경 요구사항](#환경-요구사항)
2. [프로젝트 구조](#프로젝트-구조)
3. [로컬 개발 환경 설정](#로컬-개발-환경-설정)
4. [Colab 실행 환경](#colab-실행-환경)
5. [데이터 관리](#데이터-관리)
6. [Docker 설정](#docker-설정)
7. [실험 추적](#실험-추적)
8. [모델 체크포인트 관리](#모델-체크포인트-관리)
9. [Git 브랜치 전략](#git-브랜치-전략)
10. [개발 워크플로우](#개발-워크플로우)
11. [트러블슈팅](#트러블슈팅)

---

## 환경 요구사항

### Python 환경
- **Python**: 3.10 이상 (3.10 권장)
- **패키지 관리자**: Conda (권장) 또는 venv

### GPU/CUDA (Colab/제출 환경)
- **CUDA**: 11.8 ~ 12.6
- **PyTorch**: 2.5.0 권장
- **추론 환경**: L40S GPU (48GB VRAM)
- **학습 환경**: 48GB VRAM 내에서 작동 필수

### 로컬 개발 도구
- **Git**: 버전 관리
- **VS Code**: 코드 에디터 (권장)
- **Claude Code CLI**: AI 코딩 어시스턴트

### OS 지원
- Windows 10/11
- macOS
- Linux (Ubuntu 20.04+)

---

## 프로젝트 구조

### 전체 디렉토리 구조 (대회 제출 기준)

```
HAI_Deepfake/
├── model/                       # 필수: 최종 모델 가중치
│   └── model.pt                 # 단일 모델 weight
│
├── src/                         # 희망: 모듈 분리
│   ├── models.py                # 모델 정의
│   ├── dataset.py               # 데이터 로더/전처리
│   └── utils.py                 # 공통 유틸 함수
│
├── config/                      # 필수: 설정 파일
│   └── config.yaml              # 하이퍼파라미터, 경로 등
│
├── env/                         # 필수: 환경 설정
│   ├── Dockerfile               # Docker 이미지 재현용
│   ├── requirements.txt         # Python 라이브러리 목록
│   └── environment.yml          # Conda 환경 정의
│
├── train_data/                  # 필수: 학습 데이터
│   └── [학습 데이터 + 출처/라이선스 정보]
│
├── test_data/                   # 필수: 평가 데이터
│   └── [대회 제공 평가 데이터셋]
│
├── notebooks/                   # 개발용: Jupyter 노트북
│   ├── train.ipynb              # Colab 학습용
│   ├── inference.ipynb          # Colab 추론용
│   └── eda.ipynb                # 데이터 분석용
│
├── checkpoints/                 # 학습 중 체크포인트
│   └── [epoch별 모델 저장]
│
├── submissions/                 # 제출 파일
│   └── submission_*.csv
│
├── train.py                     # 필수: 학습 엔트리 포인트
├── inference.py                 # 필수: 추론 엔트리 포인트
├── eval.py                      # 희망: 검증용 평가 코드
│
├── README.md                    # 필수: 프로젝트 설명
├── SETUP.md                     # 개발환경 설정 가이드 (본 문서)
├── Rule.md                      # 대회 규칙
├── baseline.ipynb               # Baseline 코드
└── .gitignore                   # Git 제외 파일
```

### 파일 형식 전략

**개발/실험 단계**: `.ipynb` (Jupyter Notebook)
- Colab에서 바로 실행
- 시각화 및 인터랙티브 개발
- 빠른 프로토타이핑

**최종 제출용**: `.py` (Python 스크립트)
- Docker 환경에서 실행
- 대회 규칙 준수
- 재현성 보장

---

## 로컬 개발 환경 설정

### 1. Git 저장소 클론

```bash
# GitHub 저장소 클론
git clone https://github.com/CBottle/HAI_Deepfake.git
cd HAI_Deepfake
```

### 2. Conda 가상환경 생성

```bash
# Conda 환경 생성 (Python 3.10)
conda create -n deepfake python=3.10 -y
conda activate deepfake
```

### 3. 의존성 설치

```bash
# PyTorch 설치 (CUDA 12.1 기준, 로컬 환경에 맞게 조정)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 기타 라이브러리 설치
pip install -r env/requirements.txt
```

**requirements.txt 예시**:
```txt
numpy>=1.24.0
pandas>=2.0.0
opencv-python>=4.8.0
Pillow>=10.0.0
transformers>=4.35.0
scikit-learn>=1.3.0
tqdm>=4.65.0
PyYAML>=6.0
wandb>=0.16.0
```

### 4. 로컬 테스트 (CPU)

```bash
# 작은 데이터셋으로 코드 테스트
python train.py --debug --epochs 1 --batch_size 2 --device cpu
```

---

## Colab 실행 환경

### 1. GitHub 저장소 연동

**Colab 노트북 첫 셀**:
```python
# GitHub 저장소 클론
!git clone https://github.com/CBottle/HAI_Deepfake.git
%cd HAI_Deepfake

# 최신 코드 업데이트 (이미 클론한 경우)
!git pull origin main
```

### 2. Google Drive 마운트 (데이터용)

```python
from google.colab import drive
drive.mount('/content/drive')

# 데이터 경로 심볼릭 링크
!ln -s /content/drive/MyDrive/HAI_Deepfake/train_data ./train_data
!ln -s /content/drive/MyDrive/HAI_Deepfake/test_data ./test_data
```

### 3. 환경 설정

```python
# 필요한 라이브러리 설치
!pip install -r env/requirements.txt

# 프로젝트 모듈 경로 추가
import sys
sys.path.append('/content/HAI_Deepfake/src')
```

### 4. 학습 실행

```python
# train.ipynb에서 실행
!python train.py --config config/config.yaml --device cuda
```

### 5. 추론 실행

```python
# inference.ipynb에서 실행
!python inference.py --model model/model.pt --test_dir test_data --output submissions/
```

### 6. 결과 저장 (Drive로)

```python
# 학습된 모델을 Drive에 저장
!cp model/model.pt /content/drive/MyDrive/HAI_Deepfake/model/
!cp -r checkpoints /content/drive/MyDrive/HAI_Deepfake/
```

---

## 데이터 관리

### 데이터 저장 위치

**1. Google Drive** (권장)
```
Google Drive/
└── HAI_Deepfake/
    ├── train_data/           # 학습 데이터 (10GB ~ 100GB+)
    │   ├── real/
    │   └── fake/
    └── test_data/            # 대회 제공 평가 데이터
```

**장점**:
- Colab과 연동 쉬움
- 15GB 무료 (유료 확장 가능)

**단점**:
- 대용량 데이터 업로드 시간 소요

---

**2. Kaggle Datasets** (공개 데이터)
```python
# Colab에서 Kaggle API 사용
!pip install kaggle

# Kaggle API 토큰 업로드 (~/.kaggle/kaggle.json)
!mkdir -p ~/.kaggle
!cp /content/drive/MyDrive/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# 데이터셋 다운로드
!kaggle datasets download -d [dataset-name]
!unzip [dataset-name].zip -d train_data/
```

**장점**:
- 빠른 다운로드
- 공개 데이터셋 활용 용이

---

### 데이터셋 출처 문서화

**train_data/README.md** 생성 (대회 제출 시 필수):
```markdown
# 학습 데이터 출처

## 데이터셋 목록
1. FaceForensics++ (https://github.com/ondyari/FaceForensics)
   - 라이선스: [라이선스 정보]
   - 사용량: 10,000장

2. Celeb-DF (http://www.cs.albany.edu/~lsw/celeb-deepfakeforensics.html)
   - 라이선스: [라이선스 정보]
   - 사용량: 5,000장

## 데이터 전처리
- 프레임 추출: 10 fps
- 이미지 크기: 224x224
- 증강: RandomHorizontalFlip, ColorJitter
```

---

## Docker 설정

### Dockerfile 작성 (env/Dockerfile)

```dockerfile
# Base image (대회 환경 기준)
FROM pytorch/pytorch:2.5.0-cuda12.1-cudnn9-runtime

# 작업 디렉토리
WORKDIR /workspace

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 설치
COPY env/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 프로젝트 파일 복사
COPY . .

# 기본 명령어
CMD ["python", "inference.py"]
```

### Docker 이미지 빌드 및 테스트

```bash
# Docker 이미지 빌드
docker build -t hai-deepfake:latest -f env/Dockerfile .

# 로컬에서 테스트
docker run --gpus all -v $(pwd)/test_data:/workspace/test_data \
    hai-deepfake:latest python inference.py --test_dir test_data
```

### requirements.txt 최종 확인

```bash
# 현재 환경의 패키지 목록 저장
pip freeze > env/requirements.txt

# 불필요한 패키지 제거 (수동 편집)
# Docker 이미지 크기 최적화
```

---

## 실험 추적

### Weights & Biases (wandb) 설정

**1. 설치 및 로그인**:
```bash
pip install wandb
wandb login
```

**2. 학습 코드에 통합**:
```python
import wandb

# 실험 시작
wandb.init(
    project="hai-deepfake",
    name="vit-base-exp1",
    config={
        "learning_rate": 1e-4,
        "batch_size": 32,
        "epochs": 50,
        "model": "ViT-Base"
    }
)

# 학습 중 로깅
for epoch in range(epochs):
    train_loss = train_epoch(...)
    val_auc = validate(...)

    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_auc": val_auc
    })

# 실험 종료
wandb.finish()
```

**3. 실험 비교**:
- wandb.ai에서 여러 실험 결과 시각화
- 하이퍼파라미터별 성능 비교

---

### TensorBoard (대안)

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/exp1')

for epoch in range(epochs):
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('AUC/val', val_auc, epoch)

writer.close()
```

```bash
# TensorBoard 실행
tensorboard --logdir=runs
```

---

## 모델 체크포인트 관리

### 체크포인트 저장 전략

```python
import torch
import os

def save_checkpoint(model, optimizer, epoch, val_auc, checkpoint_dir='checkpoints'):
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_auc': val_auc
    }

    # Epoch별 체크포인트
    checkpoint_path = f'{checkpoint_dir}/checkpoint_epoch_{epoch:03d}.pt'
    torch.save(checkpoint, checkpoint_path)

    # 최고 성능 모델 저장
    best_path = f'{checkpoint_dir}/best_model.pt'
    if not os.path.exists(best_path) or val_auc > get_best_auc(best_path):
        torch.save(checkpoint, best_path)
        print(f'Best model updated! AUC: {val_auc:.4f}')

def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['val_auc']
```

### 체크포인트 디렉토리 구조

```
checkpoints/
├── checkpoint_epoch_001.pt
├── checkpoint_epoch_002.pt
├── ...
├── checkpoint_epoch_050.pt
└── best_model.pt              # 최고 성능 모델
```

### 디스크 공간 관리

```python
# 최근 N개 체크포인트만 유지
def cleanup_checkpoints(checkpoint_dir, keep_last=5):
    checkpoints = sorted(glob.glob(f'{checkpoint_dir}/checkpoint_epoch_*.pt'))
    if len(checkpoints) > keep_last:
        for old_ckpt in checkpoints[:-keep_last]:
            os.remove(old_ckpt)
            print(f'Removed old checkpoint: {old_ckpt}')
```

---

## Git 브랜치 전략

### 브랜치 구조

```
main                    # 안정 버전 (제출 가능한 코드)
  ├── develop           # 개발 통합 브랜치
  │   ├── feature/vit-large        # 새 모델 실험
  │   ├── feature/data-augment     # 데이터 증강 실험
  │   └── feature/ensemble         # 앙상블 실험 (규칙 위반 시 삭제)
  └── hotfix/inference-bug         # 긴급 버그 수정
```

### 브랜치 사용 규칙

**1. main 브랜치**:
- 항상 실행 가능한 상태 유지
- 제출 가능한 코드만 merge
- Pull Request 필수

**2. develop 브랜치**:
- 실험 통합용
- 매일 작업 내용 merge

**3. feature 브랜치**:
- 새로운 실험/기능 개발
- 네이밍: `feature/[기능명]`

**4. hotfix 브랜치**:
- 긴급 버그 수정
- 네이밍: `hotfix/[버그명]`

---

### Git 워크플로우

```bash
# 1. 새 실험 시작
git checkout develop
git pull origin develop
git checkout -b feature/vit-large

# 2. 작업 및 커밋
git add .
git commit -m "Add ViT-Large model implementation"

# 3. 원격 저장소에 푸시
git push origin feature/vit-large

# 4. Pull Request 생성 (GitHub 웹)
# develop <- feature/vit-large

# 5. 실험 성공 시 develop에 merge
git checkout develop
git merge feature/vit-large

# 6. 검증 완료 후 main에 merge
git checkout main
git merge develop
git push origin main
```

---

## 개발 워크플로우

### 전체 개발 사이클

```
1. 로컬 개발 (Claude CLI 사용)
   ├── src/ 모듈 작성/수정
   ├── notebooks/ 실험 코드 작성
   └── config/ 하이퍼파라미터 조정
           ↓
2. Git 커밋 및 푸시
   ├── git add .
   ├── git commit -m "메시지"
   └── git push origin [브랜치]
           ↓
3. Colab에서 실행
   ├── git pull
   ├── 학습 실행 (train.ipynb)
   ├── 체크포인트 저장 (Drive)
   └── wandb로 결과 추적
           ↓
4. 결과 분석 및 개선
   ├── wandb에서 실험 비교
   ├── 베스트 모델 선정
   └── 다음 실험 계획
           ↓
5. 최종 제출 준비
   ├── .ipynb → .py 변환
   ├── Docker 테스트
   └── 제출 파일 생성
```

---

### 일일 작업 루틴

**아침**:
```bash
# 1. 최신 코드 동기화
git pull origin develop

# 2. 새 실험 브랜치 생성
git checkout -b feature/new-experiment
```

**작업 중**:
```bash
# 3. 로컬에서 코드 작성 (Claude CLI)
# 4. 작은 테스트 (CPU)
python train.py --debug

# 5. 커밋 (자주)
git add .
git commit -m "WIP: Add new augmentation"
git push origin feature/new-experiment
```

**저녁**:
```bash
# 6. Colab에서 본격 학습
# 7. 결과 확인 후 develop에 merge
git checkout develop
git merge feature/new-experiment
git push origin develop
```

---

## 트러블슈팅

### 1. Colab 세션 끊김

**문제**: 학습 중 세션 종료
**해결**:
- 체크포인트에서 재개:
  ```python
  if os.path.exists('checkpoints/best_model.pt'):
      start_epoch, _ = load_checkpoint(model, optimizer, 'checkpoints/best_model.pt')
  else:
      start_epoch = 0
  ```
- Colab Pro 고려

---

### 2. CUDA Out of Memory

**문제**: GPU 메모리 부족
**해결**:
- 배치 사이즈 감소:
  ```yaml
  batch_size: 16  # 32 → 16
  ```
- Gradient Accumulation:
  ```python
  accumulation_steps = 4
  for i, batch in enumerate(dataloader):
      loss = model(batch) / accumulation_steps
      loss.backward()
      if (i + 1) % accumulation_steps == 0:
          optimizer.step()
          optimizer.zero_grad()
  ```
- Mixed Precision Training:
  ```python
  from torch.cuda.amp import autocast, GradScaler
  scaler = GradScaler()

  with autocast():
      loss = model(batch)
  scaler.scale(loss).backward()
  scaler.step(optimizer)
  scaler.update()
  ```

---

### 3. Git 충돌

**문제**: merge 시 충돌 발생
**해결**:
```bash
# 충돌 파일 확인
git status

# 충돌 해결 후
git add [해결된 파일]
git commit -m "Resolve merge conflict"
```

---

### 4. 모듈 Import 오류 (Colab)

**문제**: `ModuleNotFoundError: No module named 'src'`
**해결**:
```python
import sys
sys.path.append('/content/HAI_Deepfake')

# 또는 상대 경로 사용
from src.models import MyModel
```

---

### 5. Docker 빌드 실패

**문제**: Docker 이미지 빌드 중 오류
**해결**:
```bash
# 캐시 없이 빌드
docker build --no-cache -t hai-deepfake:latest -f env/Dockerfile .

# 빌드 로그 자세히 보기
docker build --progress=plain -t hai-deepfake:latest -f env/Dockerfile .
```

---

### 6. 추론 시간 초과 (60분 제한)

**문제**: 추론이 60분을 초과
**해결**:
- 배치 추론 최적화:
  ```python
  # 배치 크기 증가
  batch_size = 64  # 메모리 허용 범위 내

  # DataLoader num_workers 증가
  dataloader = DataLoader(dataset, batch_size=64, num_workers=4)
  ```
- TorchScript 사용:
  ```python
  model = torch.jit.script(model)
  ```
- FP16 추론:
  ```python
  model.half()
  inputs = inputs.half()
  ```

---

### 7. wandb 로그인 오류 (Colab)

**문제**: Colab에서 wandb 로그인 안 됨
**해결**:
```python
# API 키 직접 입력
import wandb
wandb.login(key='your-api-key')

# 또는 환경변수 설정
import os
os.environ['WANDB_API_KEY'] = 'your-api-key'
```

---

## 참고 자료

- **대회 규칙**: [Rule.md](Rule.md)
- **Baseline 코드**: [baseline.ipynb](baseline.ipynb)
- **PyTorch 문서**: https://pytorch.org/docs/stable/index.html
- **Transformers 문서**: https://huggingface.co/docs/transformers
- **wandb 문서**: https://docs.wandb.ai/
- **Docker 문서**: https://docs.docker.com/

---

## 버전 정보

- **문서 버전**: 1.0.0
- **최종 수정일**: 2026-01-02
- **작성자**: HAI Deepfake Team
