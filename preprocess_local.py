import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from retinaface import RetinaFace

def crop_face_retina(image, bbox=None):
    """RetinaFace를 이용한 정밀 얼굴 크롭 (bbox가 있으면 바로 크롭)"""
    h, w = image.shape[:2]
    
    # 1. BBox가 없으면 감지 시도
    if bbox is None:
        # 속도 최적화를 위해 이미지 리사이즈 (Max 1024px)
        target_size = 1024
        scale = 1.0
        if max(h, w) > target_size:
            scale = target_size / max(h, w)
            small_image = cv2.resize(image, (int(w * scale), int(h * scale)))
        else:
            small_image = image

        try:
            resp = RetinaFace.detect_faces(small_image)
        except:
            resp = None
        
        if not resp or not isinstance(resp, dict):
            # 얼굴 못 찾으면 중앙 70% 크롭 (Fallback)
            center_x, center_y = w // 2, h // 2
            crop_w, crop_h = int(w * 0.7), int(h * 0.7)
            start_x = max(0, center_x - crop_w // 2)
            start_y = max(0, center_y - crop_h // 2)
            return image[start_y:start_y+crop_h, start_x:start_x+crop_w]

        max_area = 0
        best_face = None
        
        for key in resp:
            face = resp[key]
            # 좌표 복원 (Scale 역산)
            facial_area = [int(coord / scale) for coord in face['facial_area']]
            area = (facial_area[2] - facial_area[0]) * (facial_area[3] - facial_area[1])
            if area > max_area:
                max_area = area
                best_face = facial_area
        
        bbox = best_face

    # 2. BBox 기준으로 크롭 (Margin 10%)
    if bbox:
        x1, y1, x2, y2 = bbox
        
        w_face, h_face = x2 - x1, y2 - y1
        margin = 0.1
        x1 = max(0, int(x1 - w_face * margin))
        y1 = max(0, int(y1 - h_face * margin))
        x2 = min(image.shape[1], int(x2 + w_face * margin))
        y2 = min(image.shape[0], int(y2 + h_face * margin))
        
        # 정사각형 만들기
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        max_len = max(x2 - x1, y2 - y1)
        half_len = max_len // 2
        
        x1_new = max(0, cx - half_len)
        y1_new = max(0, cy - half_len)
        x2_new = min(image.shape[1], cx + half_len)
        y2_new = min(image.shape[0], cy + half_len)
        
        if x2_new - x1_new > 20 and y2_new - y1_new > 20:
                return image[y1_new:y2_new, x1_new:x2_new]
    
    return image

def process_video(video_path, output_dir, num_frames=30):
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0: cap.release(); return

    candidates = []
    scan_points = np.linspace(0, total_frames - 1, 15, dtype=int)

    for idx in scan_points:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret: continue
        
        h, w = frame.shape[:2]
        scale = min(1.0, 640 / w) # 가로 640px 정도로 줄여서 검출
        if scale < 1.0:
            small_frame = cv2.resize(frame, None, fx=scale, fy=scale)
        else:
            small_frame = frame
        
        small_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        face_area = 0
        bbox = None # 감지된 얼굴 좌표 저장
        
        try:
            resp = RetinaFace.detect_faces(small_rgb)
            if resp and isinstance(resp, dict):
                max_area = 0
                for key in resp:
                    face = resp[key]
                    # 작은 이미지 기준 좌표
                    f_area = face['facial_area']
                    area = (f_area[2] - f_area[0]) * (f_area[3] - f_area[1])
                    if area > max_area:
                        max_area = area
                        # 원본 좌표로 환산해서 저장!
                        bbox = [int(coord / scale) for coord in f_area]
                face_area = max_area
        except:
            pass
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        candidates.append((frame_rgb, face_area, bbox))

    cap.release()

    # 얼굴 크기순 정렬 -> 상위 N개 선택
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    best_frames = []
    # (frame, area, bbox) 튜플에서 필요한 것만 뽑음
    for item in candidates[:num_frames]:
        best_frames.append(item) # (frame, area, bbox) 통째로 저장
        
    # 부족하면 채우기
    while len(best_frames) < num_frames and len(best_frames) > 0:
        best_frames.append(best_frames[0])

    # 저장
    video_name = video_path.name
    save_dir = output_dir / (video_name + "_frames")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for i, (frame, area, bbox) in enumerate(best_frames):
        # Crop & Save (이미 찾은 bbox 사용 -> 중복 감지 제거!)
        cropped = crop_face_retina(frame, bbox=bbox)
        img = Image.fromarray(cropped)
        img.save(save_dir / f"frame_{i:03d}.jpg", quality=95)

def process_image(image_path, output_dir):
    img = cv2.imread(str(image_path))
    if img is None: return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    cropped = crop_face_retina(img_rgb)
    img_pil = Image.fromarray(cropped)
    
    img_pil.save(output_dir / image_path.name, quality=95)

def main():
    input_dir = Path("HAI_Deepfake/test_data")
    # 저장 공간 확보를 위해 D 드라이브 경로로 변경 (하위 폴더 포함)
    output_dir = Path(r"D:\deepfake_Data\test_frame")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files = sorted(list(input_dir.glob('*')))
    print(f"🚀 Processing {len(files)} files...")
    
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.jfif'}
    video_exts = {'.mp4', '.avi', '.mov', '.mkv'}
    
    for f in tqdm(files):
        ext = f.suffix.lower()
        if ext in video_exts:
            process_video(f, output_dir)
        elif ext in image_exts:
            process_image(f, output_dir)
            
    print("✅ All done! Zip the 'processed_test_data' folder and upload to Colab/Kaggle.")

if __name__ == '__main__':
    main()
