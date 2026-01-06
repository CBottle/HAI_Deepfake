"""
소규모 데이터셋 생성 스크립트

대규모 데이터셋에서 작은 서브셋을 만들어 빠르게 테스트/실험합니다.

사용법:
    # 1000개씩 샘플링 (Real 1000, Fake 1000)
    python scripts/create_small_dataset.py --input train_data --output train_data_small --num-samples 1000

    # 비율로 샘플링 (전체의 10%)
    python scripts/create_small_dataset.py --input train_data --output train_data_small --ratio 0.1
"""

import argparse
import shutil
import random
from pathlib import Path
from typing import Optional
from tqdm import tqdm


class SmallDatasetCreator:
    """소규모 데이터셋 생성기"""

    def __init__(self, seed: int = 42):
        """
        Args:
            seed: 랜덤 시드
        """
        self.seed = seed
        random.seed(seed)

    def create_small_dataset(
        self,
        input_dir: Path,
        output_dir: Path,
        num_samples: Optional[int] = None,
        ratio: Optional[float] = None,
        stratified: bool = True,
    ):
        """
        소규모 데이터셋 생성

        Args:
            input_dir: 원본 데이터 디렉토리
            output_dir: 출력 디렉토리
            num_samples: 클래스당 샘플 수 (None이면 ratio 사용)
            ratio: 샘플링 비율 (0.0 ~ 1.0)
            stratified: 클래스별 균등 샘플링 여부
        """
        if num_samples is None and ratio is None:
            raise ValueError("num_samples 또는 ratio 중 하나를 지정해야 합니다.")

        print("🎲 소규모 데이터셋 생성 시작")
        print(f"📂 입력: {input_dir}")
        print(f"📁 출력: {output_dir}")
        print("-" * 70)

        # 출력 디렉토리 생성
        output_dir.mkdir(parents=True, exist_ok=True)

        total_copied = 0

        # real, fake 각각 처리
        for label in ["real", "fake"]:
            label_input_dir = input_dir / label
            label_output_dir = output_dir / label

            if not label_input_dir.exists():
                print(f"⚠️  {label} 폴더를 찾을 수 없습니다: {label_input_dir}")
                continue

            # 이미지 파일 수집
            image_files = list(label_input_dir.glob("*.jpg"))
            image_files += list(label_input_dir.glob("*.png"))
            image_files += list(label_input_dir.glob("*.jpeg"))

            if len(image_files) == 0:
                print(f"⚠️  {label} 이미지를 찾을 수 없습니다.")
                continue

            # 샘플 수 결정
            if num_samples is not None:
                n_samples = min(num_samples, len(image_files))
            else:
                n_samples = int(len(image_files) * ratio)

            # 랜덤 샘플링
            sampled_files = random.sample(image_files, n_samples)

            print(f"\n📸 {label.upper()}")
            print(f"   전체: {len(image_files):,d} 이미지")
            print(
                f"   샘플: {n_samples:,d} 이미지 ({n_samples / len(image_files) * 100:.1f}%)"
            )

            # 출력 디렉토리 생성
            label_output_dir.mkdir(parents=True, exist_ok=True)

            # 파일 복사
            for src_file in tqdm(sampled_files, desc=f"Copying {label}"):
                dst_file = label_output_dir / src_file.name
                shutil.copy2(src_file, dst_file)

            total_copied += n_samples

        # 요약
        print("\n" + "=" * 70)
        print("✅ 소규모 데이터셋 생성 완료!")
        print("=" * 70)
        print(f"📊 총 복사: {total_copied:,d} 이미지")
        print(f"📁 저장 위치: {output_dir.absolute()}")

        # 데이터셋 정보 출력
        self.print_dataset_info(output_dir)

    def print_dataset_info(self, data_dir: Path):
        """데이터셋 정보 출력"""
        print("\n" + "-" * 70)
        print("📊 데이터셋 구성")
        print("-" * 70)

        for label in ["real", "fake"]:
            label_dir = data_dir / label
            if label_dir.exists():
                images = list(label_dir.glob("*.jpg"))
                images += list(label_dir.glob("*.png"))
                images += list(label_dir.glob("*.jpeg"))
                print(f"  {label.upper():5s}: {len(images):6,d} 이미지")

        print("-" * 70)


def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description="소규모 데이터셋 생성")

    parser.add_argument("--input", type=str, required=True, help="입력 데이터 디렉토리")

    parser.add_argument(
        "--output", type=str, required=True, help="출력 데이터 디렉토리"
    )

    parser.add_argument(
        "--num-samples", type=int, default=None, help="클래스당 샘플 수 (예: 1000)"
    )

    parser.add_argument(
        "--ratio", type=float, default=None, help="샘플링 비율 (예: 0.1 = 10%%)"
    )

    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드 (기본: 42)")

    return parser.parse_args()


def main():
    """메인 함수"""
    args = parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"❌ 입력 디렉토리를 찾을 수 없습니다: {input_dir}")
        return

    # 생성기 초기화
    creator = SmallDatasetCreator(seed=args.seed)

    # 데이터셋 생성
    creator.create_small_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        num_samples=args.num_samples,
        ratio=args.ratio,
    )


if __name__ == "__main__":
    main()
