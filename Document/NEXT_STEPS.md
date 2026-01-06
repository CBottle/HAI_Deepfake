# 🚀 HAI Deepfake - 다음 단계 가이드

**작성일**: 2026-01-06  
**상태**: 데이터 준비 시스템 구축 완료 ✅

---

## ✅ 완료된 작업

### 1. 스크립트 생성
- ✅ `scripts/download_datasets.py` - Kaggle 데이터 자동 다운로드
- ✅ `scripts/extract_frames.py` - 비디오 → 이미지 변환
- ✅ `scripts/create_small_dataset.py` - 소규모 데이터셋 생성

### 2. Colab 노트북
- ✅ `notebooks/data_preparation_colab.ipynb` - 데이터 준비 전용 노트북
- ✅ Google Drive 통합
- ✅ 단계별 가이드 포함

### 3. 문서
- ✅ `Document/DATA_GUIDE.md` - 상세 데이터 가이드
- ✅ `AGENTS.md` - AI 에이전트용 코딩 가이드

---

## 🎯 지금 바로 해야 할 일 (우선순위)

### 📍 Phase 1: 데이터 수집 (오늘~내일)

#### Step 1: Kaggle API 설정 (5분)
```
1. kaggle.json 다운로드
   - https://www.kaggle.com/settings
   - "Create New API Token" 클릭

2. Google Drive에 업로드
   - 위치: /MyDrive/HAI_Deepfake/kaggle.json
```

#### Step 2: Colab에서 데이터 다운로드 (30분~1시간)
```
1. Google Colab 접속
   - https://colab.research.google.com

2. 노트북 열기
   - GitHub → https://github.com/CBottle/HAI_Deepfake
   - notebooks/data_preparation_colab.ipynb

3. 런타임 설정
   - 런타임 > 런타임 유형 변경 > GPU (T4)

4. 셀 순서대로 실행
   - Step 1~8 모두 실행
   - FaceForensics++ 다운로드 (주석 해제)
```

#### Step 3: 데이터 전처리 (1~2시간)
```
1. 비디오 → 이미지 변환
   - 자동 실행됨 (노트북 Step 6)
   
2. 소규모 데이터셋 생성
   - Real 1,000장 + Fake 1,000장
   - 자동 실행됨 (노트북 Step 7)
```

**예상 소요 시간**: 총 2~3시간
**필요한 것**: Kaggle 계정, Google Drive 20GB+

---

### 📍 Phase 2: 소규모 학습 (내일~모레)

#### Step 4: 첫 학습 실행 (30분)
```
1. train_colab.ipynb 열기

2. 데이터 경로 설정
   DATA_DIR = "/content/drive/MyDrive/HAI_Deepfake/train_data_small"

3. 학습 시작
   - epochs: 10 (빠른 테스트)
   - batch_size: 16
   - 예상 시간: 20~30분
```

#### Step 5: 결과 확인
```
1. ROC-AUC 점수 확인
   - 목표: 0.7 이상

2. 체크포인트 확인
   - Google Drive/HAI_Deepfake/checkpoints/

3. 학습 곡선 분석
   - Train/Val Loss 그래프
   - 과적합 여부 체크
```

#### Step 6: 문제 해결 및 개선
```
문제가 있다면:
1. 데이터 검증 (샘플 이미지 확인)
2. 하이퍼파라미터 조정
3. 다시 학습
```

**예상 소요 시간**: 1일
**목표**: 기본 모델 작동 확인

---

### 📍 Phase 3: 전체 데이터 학습 (3일~1주)

#### Step 7: 더 많은 데이터 수집
```
옵션 A: CelebA 추가 (Real 이미지 보강)
   - Colab에서 다운로드
   - 200K 이미지 추가

옵션 B: 더 많은 Fake 데이터
   - DFDC 일부 다운로드
   - 또는 직접 생성 (DeepFaceLab)
```

#### Step 8: 전체 데이터로 학습
```
1. 데이터 경로 변경
   DATA_DIR = "/content/drive/MyDrive/HAI_Deepfake/train_data"

2. 학습 설정
   - epochs: 50
   - batch_size: 32
   - 예상 시간: 6~12시간

3. 중간 저장 확인
   - 5 epoch마다 체크포인트 저장
```

**예상 소요 시간**: 3~7일
**목표**: ROC-AUC 0.85+ 달성

---

### 📍 Phase 4: 모델 개선 (1~2주)

#### Step 9: 하이퍼파라미터 튜닝
```
실험할 것들:
1. Learning Rate
   - 1e-4 (기본)
   - 5e-5 (작게)
   - 2e-4 (크게)

2. Data Augmentation
   - Horizontal Flip
   - Rotation
   - Color Jitter
   - Cutout/Mixup

3. 모델 아키텍처
   - ViT-Base (기본)
   - ViT-Large (더 큰 모델)
   - 사전학습 딥페이크 모델
```

#### Step 10: 검증 전략
```
1. K-Fold Cross Validation
2. Stratified Split
3. 다양한 딥페이크 기법별 성능 측정
```

**예상 소요 시간**: 1~2주
**목표**: ROC-AUC 0.90+ 달성

---

### 📍 Phase 5: 제출 준비 (3~5일)

#### Step 11: 추론 코드 최적화
```
1. 60분 제한 맞추기
   - 배치 크기 조정
   - Mixed Precision
   - 불필요한 연산 제거

2. 테스트
   - 로컬에서 시간 측정
   - Colab에서 검증
```

#### Step 12: Docker 환경 구축
```
1. Dockerfile 테스트
   - 로컬에서 빌드
   - 추론 실행 확인

2. 제출 패키지 준비
   - model/best_model.pt
   - 모든 스크립트
   - requirements.txt
```

#### Step 13: 발표자료 작성
```
1. PDF 작성 (PPT 금지)
2. 내용:
   - 데이터 수집 방법
   - 모델 아키텍처
   - 학습 전략
   - 성능 개선 방법
   - 실제 서비스 적용 가능성
```

**예상 소요 시간**: 3~5일
**목표**: 제출 준비 완료

---

## 📅 전체 일정 (예상)

| Phase | 기간 | 주요 작업 | 산출물 |
|-------|------|-----------|--------|
| **Phase 1** | Day 1~2 | 데이터 수집 및 전처리 | train_data_small (2K 이미지) |
| **Phase 2** | Day 3~4 | 소규모 학습 | 첫 모델 (ROC-AUC 0.7+) |
| **Phase 3** | Day 5~11 | 전체 데이터 학습 | 중간 모델 (ROC-AUC 0.85+) |
| **Phase 4** | Day 12~25 | 모델 개선 | 최종 모델 (ROC-AUC 0.90+) |
| **Phase 5** | Day 26~30 | 제출 준비 | 제출 패키지 + PPT |

**전체 소요 시간**: 약 4주 (1개월)

---

## 🎯 오늘 할 일 (바로 지금!)

### ✅ 체크리스트

- [ ] Kaggle 계정 확인 (이미 있음 ✅)
- [ ] Kaggle API 토큰 다운로드 (`kaggle.json`)
- [ ] Google Drive에 토큰 업로드 (`/MyDrive/HAI_Deepfake/kaggle.json`)
- [ ] Colab에서 `data_preparation_colab.ipynb` 열기
- [ ] GPU 런타임 설정 (T4)
- [ ] 노트북 Step 1~4 실행 (환경 설정)
- [ ] FaceForensics++ 다운로드 시작 (Step 5)
- [ ] ☕ 커피 타임 (다운로드 대기 중...)
- [ ] 프레임 추출 실행 (Step 6)
- [ ] 소규모 데이터셋 생성 (Step 7)
- [ ] 데이터 검증 (Step 8)

**예상 소요 시간**: 2~3시간

---

## 💡 유용한 팁

### Colab 효율적으로 사용하기
```python
# GPU 메모리 확인
!nvidia-smi

# 런타임 정보
import psutil
print(f"RAM: {psutil.virtual_memory().total / 1e9:.1f} GB")
```

### Google Drive 용량 관리
```
1. 원본 비디오는 프레임 추출 후 삭제 가능
2. 체크포인트는 최근 5개만 유지
3. 불필요한 실험 결과 정리
```

### 시간 절약 팁
```
1. 소규모로 먼저 테스트 (실수 방지)
2. 밤사이 학습 돌리기 (Colab Pro 추천)
3. 여러 실험 병렬로 진행 (다른 Colab 세션)
```

---

## 📞 도움이 필요하면

### 문서 참고
1. `Document/DATA_GUIDE.md` - 데이터 관련 모든 것
2. `Document/SETUP.md` - 환경 설정
3. `AGENTS.md` - 코딩 가이드

### 문제 해결
1. 에러 메시지 복사
2. ChatGPT/Claude에게 질문
3. Kaggle Discussion 참고

---

## 🎉 마무리

### 준비 완료!
모든 도구와 문서가 준비되었습니다. 이제 시작만 하면 됩니다!

### 첫 걸음
```bash
1. kaggle.json 다운로드
2. Google Drive에 업로드
3. Colab 노트북 열기
4. 실행!
```

### 성공을 위한 마음가짐
- ✅ 소규모로 시작 → 점진적 확장
- ✅ 실험 → 분석 → 개선 반복
- ✅ 꾸준함이 핵심 (매일 조금씩)

---

**화이팅! 🚀**

질문이 있으면 언제든지 물어보세요!
