import os
import cv2
import numpy as np
from pathlib import Path

def create_dummy_data(base_dir='train_data', num_samples=5):
    """
    학습 테스트를 위한 더미 데이터 생성
    train_data/real 과 train_data/fake 폴더에 랜덤 노이즈 이미지를 생성합니다.
    """
    base_path = Path(base_dir)
    
    # 클래스별 폴더 정의
    classes = ['real', 'fake']
    
    print(f"🚀 더미 데이터 생성을 시작합니다... (위치: {base_path.absolute()})")

    for class_name in classes:
        # 폴더 생성 (train_data/real, train_data/fake)
        dir_path = base_path / class_name
        dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 폴더 생성: {class_name}")
        
        for i in range(num_samples):
            # 224x224 크기의 랜덤 컬러 이미지 생성 (RGB)
            # 0~255 사이의 랜덤값
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            
            # 파일 저장
            file_name = f"dummy_{class_name}_{i:03d}.jpg"
            file_path = dir_path / file_name
            
            # OpenCV는 BGR 순서이므로 RGB로 저장하려면 변환하거나 그냥 저장 (더미라 상관없음)
            cv2.imwrite(str(file_path), img)
            
    print(f"✅ 생성 완료! 총 {num_samples * 2}개의 이미지가 준비되었습니다.")
    print("이제 train.py를 실행하기 위한 준비운동이 끝났습니다.")

if __name__ == "__main__":
    create_dummy_data()
