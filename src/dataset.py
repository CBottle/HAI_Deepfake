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
    추론용 데이터셋 (레이블 없음) - Face Crop 기능 추가됨

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
        
        # MTCNN Face Detector (딥러닝 기반, 고품질 크롭)
        if MTCNN is not None:
            # GPU 사용 가능 시 GPU 할당
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.mtcnn = MTCNN(keep_all=True, device=device, margin=20, post_process=False)
        else:
            self.mtcnn = None
            print("⚠️ MTCNN not available. Face cropping will be disabled.")

    def _collect_files(self) -> List[Path]:
        """데이터 디렉토리에서 파일 목록 수집"""
        files = []
        for ext in self.IMAGE_EXTS | self.VIDEO_EXTS:
            files.extend(self.data_dir.glob(f"*{ext}"))
        return sorted(files)

    def __len__(self) -> int:
        return len(self.files)

    def _crop_face(self, image: np.ndarray) -> np.ndarray:
        """MTCNN으로 얼굴을 찾아 아주 여유로운 정사각형 크롭 (Loose Square Crop)"""
        if self.mtcnn is None:
            return image

        try:
            boxes, probs = self.mtcnn.detect(image)
        except Exception:
            boxes = None

        if boxes is not None and len(boxes) > 0:
            # 가장 큰 얼굴 선택
            best_box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            x1, y1, x2, y2 = map(int, best_box)
            
            # 얼굴 중심 계산
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            # 얼굴의 크기 (가로, 세로 중 긴 쪽 기준)
            w = x2 - x1
            h = y2 - y1
            face_size = max(w, h)
            
            # 여유 마진 축소 (10% - 얼굴이 화면에 꽉 차게)
            # 학습 데이터(DFDC 등)는 보통 얼굴 위주로 타이트하게 잘려있음
            margin_rate = 0.1
            square_len = int(face_size * (1 + 2 * margin_rate))
            
            # 정사각형 좌표 계산
            half_len = square_len // 2
            img_h, img_w, _ = image.shape
            
            x1_new = max(0, cx - half_len)
            y1_new = max(0, cy - half_len)
            x2_new = min(img_w, cx + half_len)
            y2_new = min(img_h, cy + half_len)
            
            # 유효성 검사
            if x2_new - x1_new > 30 and y2_new - y1_new > 30:
                cropped = image[y1_new:y2_new, x1_new:x2_new]
                return cropped.astype(np.uint8)
        
        # 얼굴을 못 찾은 경우: 중앙 70% 구역을 잘라냄 (배경 제거 효과)
        img_h, img_w, _ = image.shape
        center_x, center_y = img_w // 2, img_h // 2
        crop_w, crop_h = int(img_w * 0.7), int(img_h * 0.7)
        
        start_x = max(0, center_x - crop_w // 2)
        start_y = max(0, center_y - crop_h // 2)
        
        return image[start_y:start_y+crop_h, start_x:start_x+crop_w].astype(np.uint8)


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, List[Image.Image]]:
        """
        추론용 샘플 가져오기

        Returns:
            Tuple: (pixel_values, filename, frames)
        """
        file_path = self.files[idx]
        frames_rgb = self._read_frames(file_path)

        # 얼굴 크롭 및 PIL 변환
        processed_frames = []
        for f in frames_rgb:
            cropped = self._crop_face(f)
            processed_frames.append(Image.fromarray(cropped))

        if processed_frames:
            # 모든 프레임 처리
            inputs = self.processor(images=processed_frames, return_tensors="pt")
            pixel_values = inputs['pixel_values']
        else:
            # 빈 텐서
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
