"""
데이터셋 및 데이터 로더 모듈

이 모듈에서는 딥페이크 탐지를 위한 데이터 전처리 및 로딩을 담당합니다.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import ViTImageProcessor
import random


class DeepfakeDataset(Dataset):
    """
    딥페이크 탐지 데이터셋

    Args:
        data_dir: 데이터 디렉토리 경로
        processor: 이미지 프로세서
        num_frames: 비디오에서 추출할 프레임 수
        transform: 추가 데이터 증강 (Optional)
    """

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".jfif"}
    VIDEO_EXTS = {".mp4", ".mov"}

    def __init__(
        self,
        data_dir: str,
        processor: ViTImageProcessor,
        num_frames: int = 10,
        transform=None
    ):
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.num_frames = num_frames
        self.transform = transform

        # 클래스 매핑 (Real: 0, Fake: 1)
        self.class_to_idx = {"Real": 0, "Fake": 1}
        
        # 파일 목록과 레이블 수집
        self.samples = self._collect_samples()

    def _collect_samples(self) -> List[Tuple[Path, int]]:
        """데이터 디렉토리에서 (파일 경로, 레이블) 쌍 수집"""
        samples = []
        for class_name, label in self.class_to_idx.items():
            class_dir = self.data_dir / class_name
            
            # 대문자 폴더가 없으면 소문자 폴더 확인
            if not class_dir.exists():
                class_dir = self.data_dir / class_name.lower()
            
            if not class_dir.exists():
                continue
                
            for ext in self.IMAGE_EXTS | self.VIDEO_EXTS:
                for file_path in class_dir.glob(f"*{ext}"):
                    samples.append((file_path, label))
                    
        return sorted(samples, key=lambda x: x[0])

    def __len__(self) -> int:
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
            'filename': file_path.name
        }

    def _read_frames(self, file_path: Path) -> List[np.ndarray]:
        """이미지 또는 비디오에서 프레임 추출"""
        ext = file_path.suffix.lower()

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
        """비디오에서 균등하게 프레임 샘플링"""
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            return []

        # 균등 샘플링 인덱스
        if total_frames <= self.num_frames:
            indices = list(range(total_frames))
        else:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()
        return frames


class InferenceDataset(Dataset):
    """
    추론용 데이터셋 (레이블 없음)

    Args:
        data_dir: 데이터 디렉토리 경로
        processor: 이미지 프로세서
        num_frames: 비디오에서 추출할 프레임 수
    """

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".jfif"}
    VIDEO_EXTS = {".mp4", ".mov"}

    def __init__(
        self,
        data_dir: str,
        processor: ViTImageProcessor,
        num_frames: int = 10
    ):
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.num_frames = num_frames

        self.files = self._collect_files()

    def _collect_files(self) -> List[Path]:
        """데이터 디렉토리에서 파일 목록 수집"""
        files = []
        for ext in self.IMAGE_EXTS | self.VIDEO_EXTS:
            files.extend(self.data_dir.glob(f"*{ext}"))
        return sorted(files)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, List[Image.Image]]:
        """
        추론용 샘플 가져오기

        Returns:
            Tuple: (pixel_values, filename, frames)
        """
        file_path = self.files[idx]
        frames_rgb = self._read_frames(file_path)

        # PIL Image로 변환
        frames_pil = [Image.fromarray(f) for f in frames_rgb] if frames_rgb else []

        if frames_pil:
            # 모든 프레임 처리
            inputs = self.processor(images=frames_pil, return_tensors="pt")
            pixel_values = inputs['pixel_values']
        else:
            # 빈 텐서
            pixel_values = torch.zeros(1, 3, 224, 224)

        return pixel_values, file_path.name, frames_pil

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
        """비디오에서 균등하게 프레임 샘플링"""
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            return []

        if total_frames <= self.num_frames:
            indices = list(range(total_frames))
        else:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()
        return frames
