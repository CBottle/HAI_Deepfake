"""
Kaggle 데이터셋 다운로드 스크립트 (Google Colab용)

이 스크립트는 Colab에서 Kaggle 데이터를 Google Drive로 직접 다운로드합니다.

사용법 (Colab에서):
    1. kaggle.json 파일을 Google Drive에 업로드
    2. 이 스크립트를 실행
    3. 데이터가 자동으로 Google Drive에 저장됨
"""

import os
import subprocess
from pathlib import Path
from typing import Optional


class KaggleDownloader:
    """Kaggle 데이터셋 다운로더"""

    def __init__(self, drive_root: str = "/content/drive/MyDrive/HAI_Deepfake"):
        """
        Args:
            drive_root: Google Drive 루트 경로
        """
        self.drive_root = Path(drive_root)
        self.data_dir = self.drive_root / "datasets"
        self.kaggle_json_path = self.drive_root / "kaggle.json"

    def setup_kaggle_api(self) -> bool:
        """
        Kaggle API 설정

        Returns:
            설정 성공 여부
        """
        print("🔧 Kaggle API 설정 중...")

        # kaggle.json 존재 확인
        if not self.kaggle_json_path.exists():
            print(f"❌ kaggle.json 파일을 찾을 수 없습니다: {self.kaggle_json_path}")
            print("📝 다음 경로에 kaggle.json을 업로드하세요:")
            print(f"   {self.kaggle_json_path}")
            print("\n💡 Kaggle API 토큰 받는 방법:")
            print("   1. https://www.kaggle.com/settings")
            print("   2. 'Create New API Token' 클릭")
            print("   3. 다운로드된 kaggle.json을 Google Drive에 업로드")
            return False

        # ~/.kaggle 디렉토리 생성
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(exist_ok=True)

        # kaggle.json 복사
        import shutil

        shutil.copy(self.kaggle_json_path, kaggle_dir / "kaggle.json")

        # 권한 설정
        os.chmod(kaggle_dir / "kaggle.json", 0o600)

        # kaggle 설치
        subprocess.run(["pip", "install", "-q", "kaggle"], check=True)

        print("✅ Kaggle API 설정 완료!")
        return True

    def download_dataset(
        self,
        dataset_name: str,
        output_name: str,
        unzip: bool = True,
        force: bool = False,
    ) -> Optional[Path]:
        """
        Kaggle 데이터셋 다운로드

        Args:
            dataset_name: Kaggle 데이터셋 이름 (예: "sorokin/faceforensics")
            output_name: 저장할 폴더 이름
            unzip: 압축 해제 여부
            force: 기존 데이터 덮어쓰기

        Returns:
            다운로드된 데이터 경로
        """
        output_dir = self.data_dir / output_name

        # 이미 존재하면 skip
        if output_dir.exists() and not force:
            print(f"⏭️  데이터셋이 이미 존재합니다: {output_dir}")
            return output_dir

        # 디렉토리 생성
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"📥 다운로드 중: {dataset_name}")
        print(f"📁 저장 경로: {output_dir}")

        try:
            # Kaggle CLI로 다운로드
            cmd = [
                "kaggle",
                "datasets",
                "download",
                "-d",
                dataset_name,
                "-p",
                str(output_dir),
            ]

            if unzip:
                cmd.append("--unzip")

            subprocess.run(cmd, check=True)
            print(f"✅ 다운로드 완료: {output_name}")
            return output_dir

        except subprocess.CalledProcessError as e:
            print(f"❌ 다운로드 실패: {e}")
            return None

    def download_competition_data(
        self,
        competition_name: str,
        output_name: str,
        unzip: bool = True,
        force: bool = False,
    ) -> Optional[Path]:
        """
        Kaggle 대회 데이터 다운로드

        Args:
            competition_name: 대회 이름 (예: "deepfake-detection-challenge")
            output_name: 저장할 폴더 이름
            unzip: 압축 해제 여부
            force: 기존 데이터 덮어쓰기

        Returns:
            다운로드된 데이터 경로
        """
        output_dir = self.data_dir / output_name

        if output_dir.exists() and not force:
            print(f"⏭️  데이터가 이미 존재합니다: {output_dir}")
            return output_dir

        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"📥 대회 데이터 다운로드 중: {competition_name}")
        print(f"📁 저장 경로: {output_dir}")

        try:
            cmd = [
                "kaggle",
                "competitions",
                "download",
                "-c",
                competition_name,
                "-p",
                str(output_dir),
            ]

            if unzip:
                cmd.append("--unzip")

            subprocess.run(cmd, check=True)
            print(f"✅ 다운로드 완료: {output_name}")
            return output_dir

        except subprocess.CalledProcessError as e:
            print(f"❌ 다운로드 실패: {e}")
            return None

    def get_dataset_info(self) -> dict:
        """
        현재 다운로드된 데이터셋 정보 반환

        Returns:
            데이터셋 정보 딕셔너리
        """
        if not self.data_dir.exists():
            return {}

        info = {}
        for dataset_dir in self.data_dir.iterdir():
            if dataset_dir.is_dir():
                # 파일 개수 세기
                file_count = sum(1 for _ in dataset_dir.rglob("*") if _.is_file())
                # 크기 계산
                total_size = sum(
                    f.stat().st_size for f in dataset_dir.rglob("*") if f.is_file()
                )

                info[dataset_dir.name] = {
                    "path": str(dataset_dir),
                    "file_count": file_count,
                    "size_gb": round(total_size / (1024**3), 2),
                }

        return info


# 추천 데이터셋 목록
RECOMMENDED_DATASETS = {
    "소규모 (테스트용)": [
        {
            "name": "sorokin/faceforensics",
            "output": "faceforensics",
            "size": "~10GB",
            "description": "FaceForensics++ - 가장 인기있는 딥페이크 데이터셋",
        }
    ],
    "중규모": [
        {
            "name": "sorokin/faceforensics",
            "output": "faceforensics",
            "size": "~10GB",
            "description": "FaceForensics++",
        },
        {
            "name": "jessicali9530/celeba-dataset",
            "output": "celeba",
            "size": "~1.5GB",
            "description": "CelebA - Real 얼굴 이미지 200K",
        },
    ],
    "대규모 (고성능)": [
        {
            "competition": "deepfake-detection-challenge",
            "output": "dfdc",
            "size": "~470GB",
            "description": "DFDC - Facebook AI 대규모 딥페이크 데이터셋",
        }
    ],
}


def print_recommendations():
    """추천 데이터셋 출력"""
    print("\n" + "=" * 70)
    print("📊 추천 데이터셋 목록")
    print("=" * 70)

    for category, datasets in RECOMMENDED_DATASETS.items():
        print(f"\n🎯 {category}")
        print("-" * 70)
        for ds in datasets:
            if "name" in ds:
                print(f"  📦 Dataset: {ds['name']}")
            elif "competition" in ds:
                print(f"  🏆 Competition: {ds['competition']}")
            print(f"     📁 Output: {ds['output']}")
            print(f"     💾 Size: {ds['size']}")
            print(f"     📝 {ds['description']}")
            print()


def main():
    """메인 실행 함수"""
    print("🚀 HAI Deepfake - Kaggle 데이터 다운로더")
    print("=" * 70)

    # Google Drive 마운트 확인
    drive_path = Path("/content/drive/MyDrive/HAI_Deepfake")
    if not drive_path.exists():
        print("⚠️  Google Colab에서 실행하세요!")
        print("📝 Colab에서 다음 코드를 먼저 실행:")
        print("   from google.colab import drive")
        print("   drive.mount('/content/drive')")
        return

    # 다운로더 초기화
    downloader = KaggleDownloader()

    # Kaggle API 설정
    if not downloader.setup_kaggle_api():
        return

    # 추천 데이터셋 출력
    print_recommendations()

    # 사용자 선택
    print("=" * 70)
    print("💡 사용 예시:")
    print("-" * 70)
    print("# 소규모로 시작 (추천)")
    print('downloader.download_dataset("sorokin/faceforensics", "faceforensics")')
    print()
    print("# CelebA 추가 (Real 이미지)")
    print('downloader.download_dataset("jessicali9530/celeba-dataset", "celeba")')
    print()
    print("# 대규모 (주의: 470GB)")
    print(
        'downloader.download_competition_data("deepfake-detection-challenge", "dfdc")'
    )
    print("=" * 70)

    # 현재 데이터셋 정보
    info = downloader.get_dataset_info()
    if info:
        print("\n📊 현재 다운로드된 데이터셋:")
        for name, details in info.items():
            print(
                f"  ✅ {name}: {details['file_count']} files, {details['size_gb']} GB"
            )
    else:
        print("\n📭 아직 다운로드된 데이터셋이 없습니다.")

    return downloader


if __name__ == "__main__":
    downloader = main()
