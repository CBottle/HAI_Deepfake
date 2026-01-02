# HAI Deepfake Detection

> 딥페이크 탐지 AI 모델 개발 프로젝트
> HAI(하이)! - Hecto AI Challenge : 2025 하반기 헥토 채용 AI 경진대회

## 프로젝트 개요

이미지 및 동영상에서 딥페이크(AI 생성 콘텐츠)를 탐지하는 AI 모델을 개발합니다.

- **입력**: 이미지(jpg, jpeg, png, jfif) 또는 동영상(mp4, mov)
- **출력**: Real(0) 또는 Fake(1) 확률
- **평가 지표**: ROC-AUC (with Sample Weights)

## 프로젝트 구조

```
HAI_Deepfake/
├── model/                       # 최종 모델 가중치
│   └── model.pt
├── src/                         # 소스 코드 모듈
│   ├── __init__.py
│   ├── models.py                # 모델 정의
│   ├── dataset.py               # 데이터 로더/전처리
│   └── utils.py                 # 유틸리티 함수
├── config/                      # 설정 파일
│   └── config.yaml
├── env/                         # 환경 설정
│   ├── Dockerfile
│   ├── requirements.txt
│   └── environment.yml
├── notebooks/                   # Jupyter 노트북
│   ├── train.ipynb              # Colab 학습용
│   └── inference.ipynb          # Colab 추론용
├── checkpoints/                 # 학습 체크포인트
├── submissions/                 # 제출 파일
├── train.py                     # 학습 스크립트
├── inference.py                 # 추론 스크립트
├── baseline.ipynb               # Baseline 코드
├── SETUP.md                     # 개발환경 설정 가이드
└── README.md                    # 프로젝트 설명 (본 문서)
```

## 빠른 시작

### 1. 저장소 클론

```bash
git clone https://github.com/CBottle/HAI_Deepfake.git
cd HAI_Deepfake
```

### 2. 환경 설정

**Conda 환경 생성**:
```bash
conda env create -f env/environment.yml
conda activate deepfake
```

**또는 pip 사용**:
```bash
pip install -r env/requirements.txt
```

### 3. 추론 실행

```bash
python inference.py \
    --config config/config.yaml \
    --test_dir test_data \
    --output submissions/submission.csv
```

### 4. 학습 실행 (데이터 준비 후)

```bash
python train.py --config config/config.yaml
```

## Google Colab 사용법

### 추론

1. `notebooks/inference.ipynb` 열기
2. Google Colab에서 실행
3. GitHub 저장소 클론 및 환경 설정
4. 추론 실행

### 학습

1. `notebooks/train.ipynb` 열기
2. Google Colab에서 실행
3. 학습 데이터 준비 (Google Drive)
4. 학습 실행

자세한 내용은 [SETUP.md](SETUP.md)를 참조하세요.

## 모델

- **Baseline**: Vision Transformer (ViT)
  - 사전학습 모델: `prithivMLmods/Deep-Fake-Detector-v2-Model`
  - 또는 `google/vit-base-patch16-224`

## 데이터

- **학습 데이터**: 직접 수집/생성 (외부 데이터 사용 가능)
- **테스트 데이터**: 대회 제공

### 데이터 전처리

- 이미지: 224x224 패딩
- 비디오: 균등하게 10프레임 샘플링
- 각 프레임별 독립 추론 후 평균

## 대회 규칙 준수

- ✅ 이미지 단위 입력 처리
- ✅ 단일 모델 사용
- ✅ 로컬 실행 가능
- ✅ 추론 시간 60분 이내

자세한 규칙은 [Rule.md](Rule.md)를 참조하세요.

## 개발 워크플로우

```
로컬 개발 (Claude CLI)
    ↓
Git 커밋 및 푸시
    ↓
Colab에서 실행 (학습/추론)
    ↓
결과 분석 및 개선
```

## 성능 개선 아이디어

- [ ] 데이터 증강 (Augmentation)
- [ ] 다양한 모델 아키텍처 실험
- [ ] 하이퍼파라미터 튜닝
- [ ] 멀티 프레임 정보 활용 (후처리)
- [ ] 추론 속도 최적화

## 기술 스택

- **언어**: Python 3.10
- **프레임워크**: PyTorch 2.5.0
- **모델**: Transformers (Hugging Face)
- **실험 추적**: Weights & Biases
- **환경**: Google Colab, Docker

## 참고 자료

- [대회 규칙](Rule.md)
- [개발환경 설정](SETUP.md)
- [Baseline 코드](baseline.ipynb)
- [PyTorch 문서](https://pytorch.org/docs/)
- [Transformers 문서](https://huggingface.co/docs/transformers)

## 라이선스

이 프로젝트는 HAI(하이)! - Hecto AI Challenge 대회를 위한 것입니다.

## 팀 정보

- **팀명**: TBD
- **팀원**: TBD

---

**Last Updated**: 2026-01-02
