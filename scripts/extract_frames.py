"""
비디오에서 프레임 추출 스크립트

Kaggle 딥페이크 데이터셋의 비디오 파일을 이미지로 변환합니다.
train_data/real, train_data/fake 폴더 구조로 자동 정리합니다.

사용법:
    python scripts/extract_frames.py --input datasets/faceforensics --output train_data
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
import multiprocessing as mp
from functools import partial


class VideoFrameExtractor:
    """비디오 프레임 추출기"""

    VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}

    def __init__(
        self,
        max_frames: int = 30,
        sample_method: str = "uniform",
        min_face_size: int = 64,
        quality: int = 95,
    ):
        """
        Args:
            max_frames: 비디오당 최대 추출 프레임 수
            sample_method: 샘플링 방법 ('uniform', 'random', 'first')
            min_face_size: 최소 얼굴 크기 (픽셀)
            quality: JPEG 품질 (0-100)
        """
        self.max_frames = max_frames
        self.sample_method = sample_method
        self.min_face_size = min_face_size
        self.quality = quality

    def extract_frames_from_video(
        self, video_path: Path, output_dir: Path, label: str
    ) -> int:
        """
        비디오에서 프레임 추출

        Args:
            video_path: 비디오 파일 경로
            output_dir: 출력 디렉토리
            label: 레이블 ('real' or 'fake')

        Returns:
            추출된 프레임 수
        """
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            return 0

        # 샘플링 인덱스 결정
        frame_indices = self._get_frame_indices(total_frames)

        # 출력 디렉토리 생성
        label_dir = output_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)

        # 비디오 이름 (확장자 제거)
        video_name = video_path.stem

        extracted_count = 0

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()

            if not ret:
                continue

            # RGB 변환
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 프레임 저장
            output_path = label_dir / f"{video_name}_frame_{idx:04d}.jpg"
            cv2.imwrite(
                str(output_path),
                cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, self.quality],
            )
            extracted_count += 1

        cap.release()
        return extracted_count

    def _get_frame_indices(self, total_frames: int) -> List[int]:
        """
        프레임 샘플링 인덱스 계산

        Args:
            total_frames: 전체 프레임 수

        Returns:
            샘플링할 프레임 인덱스 리스트
        """
        num_frames = min(self.max_frames, total_frames)

        if self.sample_method == "uniform":
            # 균등 샘플링
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        elif self.sample_method == "random":
            # 랜덤 샘플링
            indices = np.random.choice(total_frames, num_frames, replace=False)
            indices = np.sort(indices)

        elif self.sample_method == "first":
            # 처음 N개
            indices = np.arange(num_frames)

        else:
            raise ValueError(f"Unknown sample method: {self.sample_method}")

        return indices.tolist()

    def process_dataset(
        self,
        input_dir: Path(r"D:\deepfake_Data\train"),
        output_dir: Path(r"D:\deepfake_Data\extracted_frames"),
        max_videos: Optional[int] = None,
        num_workers: int = 4,
    ):
        """
        데이터셋 전체 처리

        Args:
            input_dir: 입력 디렉토리 (real/, fake/ 하위 폴더 가정)
            output_dir: 출력 디렉토리
            max_videos: 최대 처리 비디오 수 (None이면 전체)
            num_workers: 병렬 처리 워커 수
        """
        print(f"🎬 비디오 → 이미지 변환 시작")
        print(f"📂 입력: {input_dir}")
        print(f"📁 출력: {output_dir}")
        print(f"⚙️  설정: {self.max_frames} frames/video, {self.sample_method} sampling")
        print("-" * 70)

        # real, fake 폴더 찾기
        for label in ["real", "fake"]:
            label_dir = input_dir / label

            if not label_dir.exists():
                print(f"⚠️  {label} 폴더를 찾을 수 없습니다: {label_dir}")
                continue

            # 비디오 파일 수집
            video_files = []
            for ext in self.VIDEO_EXTS:
                video_files.extend(label_dir.glob(f"*{ext}"))

            if max_videos:
                video_files = video_files[:max_videos]

            print(f"\n📹 {label.upper()} 비디오: {len(video_files)}개")

            if len(video_files) == 0:
                print(f"   ⚠️  비디오 파일이 없습니다.")
                continue

            # 프레임 추출
            total_frames = 0

            for video_path in tqdm(video_files, desc=f"Processing {label}"):
                count = self.extract_frames_from_video(video_path, output_dir, label)
                total_frames += count

            print(f"   ✅ 추출 완료: {total_frames} 프레임")

        # 결과 요약
        self.print_summary(output_dir)

    def print_summary(self, output_dir: Path):
        """결과 요약 출력"""
        print("\n" + "=" * 70)
        print("📊 데이터셋 요약")
        print("=" * 70)

        for label in ["real", "fake"]:
            label_dir = output_dir / label
            if label_dir.exists():
                images = list(label_dir.glob("*.jpg"))
                print(f"  {label.upper():5s}: {len(images):6,d} 이미지")

        print("=" * 70)


def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description="비디오에서 프레임 추출")

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="입력 디렉토리 (real/, fake/ 하위 폴더 필요)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="train_data",
        help="출력 디렉토리 (기본: train_data)",
    )

    parser.add_argument(
        "--max-frames", type=int, default=30, help="비디오당 최대 프레임 수 (기본: 30)"
    )

    parser.add_argument(
        "--sample-method",
        type=str,
        choices=["uniform", "random", "first"],
        default="uniform",
        help="샘플링 방법 (기본: uniform)",
    )

    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="최대 처리 비디오 수 (테스트용, 기본: 전체)",
    )

    parser.add_argument(
        "--quality", type=int, default=95, help="JPEG 품질 (0-100, 기본: 95)"
    )

    parser.add_argument(
        "--num-workers", type=int, default=4, help="병렬 처리 워커 수 (기본: 4)"
    )

    return parser.parse_args()


def main():
    """메인 함수"""
    args = parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"❌ 입력 디렉토리를 찾을 수 없습니다: {input_dir}")
        return

    # 추출기 초기화
    extractor = VideoFrameExtractor(
        max_frames=args.max_frames,
        sample_method=args.sample_method,
        quality=args.quality,
    )

    # 처리 시작
    extractor.process_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        max_videos=args.max_videos,
        num_workers=args.num_workers,
    )

    print("\n✅ 모든 프레임 추출 완료!")
    print(f"📁 출력 위치: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
