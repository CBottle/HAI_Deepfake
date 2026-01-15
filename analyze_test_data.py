import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

def analyze_test_data(data_dir):
    data_dir = Path(data_dir)
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    video_exts = {'.mp4', '.avi', '.mov', '.mkv'}
    
    # 통계 변수
    stats = {
        'total_files': 0,
        'image_count': 0,
        'video_count': 0,
        'resolutions': [],
        'face_ratios': [],
        'face_counts': [],
        'brightness': []
    }
    
    # OpenCV 얼굴 인식기
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    files = list(data_dir.glob('*'))
    print(f"🔍 Analyzing {len(files)} files in {data_dir}...")
    
    for f_path in tqdm(files):
        ext = f_path.suffix.lower()
        
        # 이미지 처리
        if ext in image_exts:
            stats['image_count'] += 1
            img = cv2.imread(str(f_path))
            if img is None: continue
            
            h, w = img.shape[:2]
            stats['resolutions'].append((w, h))
            stats['brightness'].append(np.mean(img))
            
            # 얼굴 감지
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            stats['face_counts'].append(len(faces))
            
            if len(faces) > 0:
                # 가장 큰 얼굴 기준 비율
                max_face = max(faces, key=lambda f: f[2] * f[3])
                face_area = max_face[2] * max_face[3]
                stats['face_ratios'].append((face_area / (w * h)) * 100)
            else:
                stats['face_ratios'].append(0.0)
                
        # 비디오 처리 (첫 프레임만)
        elif ext in video_exts:
            stats['video_count'] += 1
            cap = cv2.VideoCapture(str(f_path))
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                h, w = frame.shape[:2]
                stats['resolutions'].append((w, h))
                stats['brightness'].append(np.mean(frame))
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                stats['face_counts'].append(len(faces))
                
                if len(faces) > 0:
                    max_face = max(faces, key=lambda f: f[2] * f[3])
                    face_area = max_face[2] * max_face[3]
                    stats['face_ratios'].append((face_area / (w * h)) * 100)
                else:
                    stats['face_ratios'].append(0.0)
        
        stats['total_files'] += 1

    # 결과 리포트 출력
    print("\n" + "="*40)
    print(f"📊 Test Data Analysis Report")
    print("="*40)
    print(f"Total Files: {stats['total_files']} (Images: {stats['image_count']}, Videos: {stats['video_count']})")
    
    if stats['resolutions']:
        widths, heights = zip(*stats['resolutions'])
        print(f"\n[Resolution]")
        print(f"  Max: {max(widths)}x{max(heights)}")
        print(f"  Min: {min(widths)}x{min(heights)}")
        print(f"  Avg: {int(np.mean(widths))}x{int(np.mean(heights))}")
        
    if stats['face_ratios']:
        ratios = np.array(stats['face_ratios'])
        print(f"\n[Face Ratio (Face Area / Image Area)]")
        print(f"  Avg: {np.mean(ratios):.2f}%")
        print(f"  Max: {np.max(ratios):.2f}%")
        print(f"  Min: {np.min(ratios):.2f}%")
        print(f"  Zero Face Detected: {np.sum(ratios == 0)} files ({np.sum(ratios == 0)/len(ratios)*100:.1f}%)")
        
    print(f"\n[Brightness (0-255)]")
    print(f"  Avg: {np.mean(stats['brightness']):.1f}")
    print("="*40)

if __name__ == '__main__':
    # 로컬 테스트용 경로 (없으면 Colab 경로 사용)
    local_path = "HAI_Deepfake/test_data"
    colab_path = "/content/test_data/test_data"
    
    target_dir = local_path if os.path.exists(local_path) else colab_path
    if os.path.exists(target_dir):
        analyze_test_data(target_dir)
    else:
        print(f"❌ Data directory not found: {target_dir}")
