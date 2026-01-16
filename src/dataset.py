"""
데이터셋 및 데이터 로더 모듈

이 모듈에서는 딥페이크 탐지를 위한 데이터 전처리 및 로딩을 담당합니다.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import ViTImageProcessor
import random
from pathlib import Path
import torch
import cv2
import numpy as np
from PIL import Image

try:
    from facenet_pytorch import MTCNN
except ImportError:
    print("Warning: facenet_pytorch not installed. Install with `pip install facenet-pytorch`")
    MTCNN = None

class DeepfakeDataset(Dataset):
    """
    GRAVEX-200K (CSV 기반) 데이터셋 로더
    """
    # (기존 코드와 동일)
    IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv'}

    def __init__(
        self,
        csv_path: str,         # labels.csv 경로
        img_dir: str,          # 실제 이미지들이 모여있는 폴더 경로
        processor: ViTImageProcessor,
        transform=None,
        num_frames: int = 1    # 현재 데이터셋은 이미지이므로 1로 기본값 설정
    ):
        self.img_dir = Path(img_dir)
        self.processor = processor
        self.transform = transform
        self.num_frames = num_frames

        # 1. CSV 파일 로드
        self.data_df = pd.read_csv(csv_path)
        
        # 2. 샘플 리스트 생성 (기존 self.samples 구조 유지: [(경로, 라벨), ...])
        self.samples = []
        for _, row in self.data_df.iterrows():
            img_path = self.img_dir / row['filename']
            label = int(row['label'])
            self.samples.append((str(img_path), label))
            
        print(f"✅ {csv_path} 로드 완료: {len(self.samples)}개의 샘플")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """
        데이터셋에서 샘플 가져오기

        Returns:
            dict: {'pixel_values': Tensor, 'labels': int, 'filename': str}
        """
        file_path, label = self.samples[idx]

        # 프레임 추출
        frames = self._read_frames(file_path)

        # 이미지 전처리
        if frames:
            chosen_frame = random.choice(frames) # chosen_frame은 현재 numpy array 상태
            
            # 1. 증강 적용 단계
            if self.transform:
                # Albumentations인 경우
                if hasattr(self.transform, "is_albumentations") or "albumentations" in str(type(self.transform)).lower():
                    augmented = self.transform(image=chosen_frame)
                    image_np = augmented['image']
                    
                    # Tensor(ToTensorV2) 처리
                    if torch.is_tensor(image_np):
                        image_np = image_np.permute(1, 2, 0).cpu().numpy()
                    
                    # Float(Normalize) 처리: 0~1 사이면 255 곱해주기
                    if image_np.dtype == np.float32 or image_np.dtype == np.float64:
                        image_np = (image_np * 255).astype(np.uint8)
                    
                    image = Image.fromarray(image_np)
                
                # Torchvision인 경우
                else:
                    image_pil = Image.fromarray(chosen_frame)
                    image = self.transform(image_pil)
            else:
                # 증강 안 쓸 때
                image = Image.fromarray(chosen_frame)

            # 2. Processor를 통한 전처리 (최종)
            # image는 여기서 무조건 PIL Image 형태여야 함!
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].squeeze(0)
            
        else:
            # 에러 시 빈 이미지 처리
            pixel_values = torch.zeros(3, 224, 224)

        return {
            'pixel_values': pixel_values,
            'labels': label,
            'filename': Path(file_path).name
        }

    def _read_frames(self, file_path: Path) -> List[np.ndarray]:
        """이미지 또는 비디오에서 프레임 추출"""
        ext = Path(file_path).suffix.lower()

        # 이미지 파일
        if ext in self.IMAGE_EXTS:
            try:
                img = Image.open(file_path).convert("RGB")
                return [np.array(img)]
            except Exception:
                return []

        # 비디오 파일
        if ext in self.VIDEO_EXTS:
            return self._extract_video_frames(file_path)

        return []

    def _extract_video_frames(self, video_path: Path) -> List[np.ndarray]:
        """비디오에서 얼굴이 가장 크게/잘 나온 프레임들을 선별하여 추출"""
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            return []

        # 1. 빠른 스캔을 위한 OpenCV 얼굴 인식기 (Haarcascade)
        # (DeepfakeDataset은 학습용이라 face_cascade가 없을 수 있으므로 여기서 로드)
        if not hasattr(self, 'face_cascade') or self.face_cascade is None:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        candidates = []
        stride = max(1, total_frames // 50)  # 전체에서 최대 50장 정도만 샘플링해서 검사 (속도 최적화)

        for i in range(0, total_frames, stride):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            
            # 흑백 변환 후 얼굴 감지
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            face_area = 0
            if len(faces) > 0:
                # 가장 큰 얼굴의 면적 계산
                max_face = max(faces, key=lambda f: f[2] * f[3])
                face_area = max_face[2] * max_face[3]
            
            # (프레임(RGB), 얼굴크기) 저장
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            candidates.append((frame_rgb, face_area))

        cap.release()

        # 2. 베스트 프레임 선정 (얼굴 크기 내림차순)
        # 얼굴이 발견된 프레임 위주로 선택
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        best_frames = []
        for frame, area in candidates[:self.num_frames]:
            best_frames.append(frame)
            
        # 만약 얼굴을 하나도 못 찾았거나 프레임이 부족하면? -> 부족한 만큼 채우기 (기존 방식)
        if len(best_frames) < self.num_frames:
            # 다시 열어서 균등 샘플링으로 부족분 채움
            cap = cv2.VideoCapture(str(video_path))
            missing_count = self.num_frames - len(best_frames)
            indices = np.linspace(0, total_frames - 1, missing_count, dtype=int)
            
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if ret:
                    best_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()

        return best_frames[:self.num_frames]


class InferenceDataset(Dataset):
    """
    추론용 데이터셋 (레이블 없음)

    Args:
        data_dir: 데이터 디렉토리 경로
        transform: 이미지 변환 함수 (timm transform)
        num_frames: 비디오에서 추출할 프레임 수
    """

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".jfif"}
    VIDEO_EXTS = {".mp4", ".mov"}

    def __init__(
        self,
        data_dir: str,
        transform=None,
        num_frames: int = 10
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.num_frames = num_frames

        self.files = self._collect_files()
        
        # OpenCV Face Detector (Best Frame Selection용)
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except Exception as e:
            print(f"Warning: Failed to load OpenCV Haarcascade: {e}")
            self.face_cascade = None

    def _collect_files(self) -> List[Path]:
        """데이터 디렉토리에서 파일 목록 수집"""
        files = []
        for ext in self.IMAGE_EXTS | self.VIDEO_EXTS:
            files.extend(self.data_dir.glob(f"*{ext}"))
        return sorted(files)

    def __len__(self) -> int:
        return len(self.files)

    def _resize_with_padding(self, image: np.ndarray) -> np.ndarray:
        """비율 유지를 위한 패딩 리사이즈 (Letterbox)"""
        h, w = image.shape[:2]
        longest = max(h, w)
        
        # 정사각형 캔버스 (검은색)
        canvas = np.zeros((longest, longest, 3), dtype=np.uint8)
        
        # 중앙 배치 계산
        top = (longest - h) // 2
        left = (longest - w) // 2
        
        # 이미지 붙여넣기
        canvas[top:top+h, left:left+w] = image
        
        return canvas

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, List[Image.Image]]:
        """
        추론용 샘플 가져오기

        Returns:
            Tuple: (pixel_values, filename, frames)
        """
        file_path = self.files[idx]
        frames_rgb = self._read_frames(file_path)

        # 패딩 리사이즈 및 Transform 적용
        pixel_values_list = []
        processed_frames = [] # 디버깅용 저장 (PIL Image)
        
        for f in frames_rgb:
            # 1. 패딩 리사이즈 (Square)
            padded = self._resize_with_padding(f)
            img_pil = Image.fromarray(padded)
            processed_frames.append(img_pil)
            
            # 2. Transform 적용 (Resize -> Tensor -> Normalize)
            if self.transform:
                tensor = self.transform(img_pil)
                pixel_values_list.append(tensor)
            else:
                # Transform 없으면 기본 Tensor 변환 (비추천)
                from torchvision.transforms.functional import to_tensor
                pixel_values_list.append(to_tensor(img_pil))

        if pixel_values_list:
            # (Batch, C, H, W) 형태로 스택
            pixel_values = torch.stack(pixel_values_list)
        else:
            # 빈 텐서 (에러 방지용)
            pixel_values = torch.zeros(1, 3, 224, 224)

        return pixel_values, file_path.name, processed_frames

    def _read_frames(self, file_path: Path) -> List[np.ndarray]:
        """이미지 또는 비디오에서 프레임 추출"""
        ext = file_path.suffix.lower()

        if ext in self.IMAGE_EXTS:
            try:
                img = Image.open(file_path).convert("RGB")
                return [np.array(img)]
            except Exception:
                return []

        if ext in self.VIDEO_EXTS:
            return self._extract_video_frames(file_path)

        return []

    def _extract_video_frames(self, video_path: Path) -> List[np.ndarray]:
        """비디오에서 얼굴이 가장 크게/잘 나온 프레임들을 선별하여 추출"""
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            return []

        candidates = []
        stride = max(1, total_frames // 50)  # 전체에서 최대 50장 정도만 샘플링해서 검사

        for i in range(0, total_frames, stride):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            
            # 얼굴 크기 측정 (중요도 산정)
            face_area = 0
            if self.face_cascade is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                if len(faces) > 0:
                    max_face = max(faces, key=lambda f: f[2] * f[3])
                    face_area = max_face[2] * max_face[3]
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            candidates.append((frame_rgb, face_area))

        cap.release()

        # 얼굴 크기 내림차순 정렬 후 상위 N개 선택
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        best_frames = []
        for frame, area in candidates[:self.num_frames]:
            best_frames.append(frame)
            
        # 부족하면 균등 샘플링으로 채움
        if len(best_frames) < self.num_frames:
            cap = cv2.VideoCapture(str(video_path))
            missing_count = self.num_frames - len(best_frames)
            indices = np.linspace(0, total_frames - 1, missing_count, dtype=int)
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if ret:
                    best_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()

        return best_frames[:self.num_frames]
