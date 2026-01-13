import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import urllib.request
import os

# 1. 모델 다운로드 URL 설정
PROTO_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

def download_model():
    """모델 파일이 없으면 다운로드하고 절대 경로를 반환합니다."""
    curr_dir = Path(os.getcwd())
    proto_path = curr_dir / "deploy.prototxt"
    model_path = curr_dir / "res10_300x300_ssd_iter_140000.caffemodel"
    
    if not proto_path.exists():
        print("📥 모델 설정 파일(prototxt) 다운로드 중...")
        urllib.request.urlretrieve(PROTO_URL, str(proto_path))
    if not model_path.exists():
        print("📥 모델 가중치 파일(caffemodel) 다운로드 중...")
        urllib.request.urlretrieve(MODEL_URL, str(model_path))
    
    return str(proto_path.absolute()), str(model_path.absolute())

def extract_worker(video_info, proto_path, model_path, max_frames, sample_method, min_face_size, quality, margin):
    """각 프로세스에서 실행될 실제 크롭 로직"""
    video_path, output_root, label = video_info
    
    # 워커 내부에서 DNN 모델 로드 (절대 경로 사용)
    try:
        net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
    except Exception as e:
        return 0

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return 0

    # 샘플링 인덱스 결정
    num_to_sample = min(max_frames, total_frames)
    if sample_method == "uniform":
        frame_indices = np.linspace(0, total_frames - 1, num_to_sample, dtype=int)
    else:
        frame_indices = np.sort(np.random.choice(total_frames, num_to_sample, replace=False))

    video_name = video_path.stem
    extracted_count = 0

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret: continue

        (h, w) = frame.shape[:2]
        # 얼굴 탐지를 위한 전처리
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        best_confidence = 0
        best_box = None

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5: # 신뢰도 50% 이상만
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                if (endX - startX) > min_face_size:
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_box = (startX, startY, endX, endY)

        if best_box:
            (x1, y1, x2, y2) = best_box
            fw, fh = x2 - x1, y2 - y1
            
            # 마진 적용 (얼굴 주변 여유분)
            mx, my = int(fw * margin), int(fh * margin)
            nx1, ny1 = max(0, x1 - mx), max(0, y1 - my)
            nx2, ny2 = min(w, x2 + mx), min(h, y2 + my)

            face_crop = frame[ny1:ny2, nx1:nx2]
            
            # 저장 경로 설정
            save_dir = output_root / label
            save_dir.mkdir(parents=True, exist_ok=True)
            output_path = save_dir / f"{video_name}_idx{idx:04d}.jpg"
            
            cv2.imwrite(str(output_path), face_crop, [cv2.IMWRITE_JPEG_QUALITY, quality])
            extracted_count += 1

    cap.release()
    return extracted_count

class FaceExtractor:
    def __init__(self, max_frames=20, sample_method="uniform", min_face_size=128, quality=95, margin=0.25):
        self.max_frames = max_frames
        self.sample_method = sample_method
        self.min_face_size = min_face_size
        self.quality = quality
        self.margin = margin

    def run(self, input_path: str, output_path: str, num_workers=4):
        # 1. 모델 준비 및 절대 경로 가져오기
        proto_abs, model_abs = download_model()
        
        input_root = Path(input_path)
        output_root = Path(output_path)
        
        print(f"🚀 [Face-Crop] 작업을 시작합니다.")
        
        for label in ["real", "fake"]:
            target_dir = input_root / label
            if not target_dir.exists():
                print(f"⚠️ {label} 폴더를 찾을 수 없습니다.")
                continue

            video_files = [v for v in target_dir.glob("*") if v.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}]
            print(f"📹 {label.upper()} 비디오 개수: {len(video_files)}개")

            # 워커에 고정 인자 전달 (모델 경로 포함)
            worker_fn = partial(
                extract_worker, 
                proto_path=proto_abs,
                model_path=model_abs,
                max_frames=self.max_frames, 
                sample_method=self.sample_method,
                min_face_size=self.min_face_size, 
                quality=self.quality, 
                margin=self.margin
            )
            
            video_infos = [(v, output_root, label) for v in video_files]

            with mp.Pool(num_workers) as pool:
                results = list(tqdm(pool.imap(worker_fn, video_infos), total=len(video_infos), desc=f"Extracting {label}"))
            
            print(f"✅ {label.upper()} 완료: 총 {sum(results)}개 프레임 추출")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()

if __name__ == "__main__":
    # 윈도우 멀티프로세싱 필수 구문
    args = parse_args()
    
    extractor = FaceExtractor(max_frames=20, min_face_size=128)
    extractor.run(args.input, args.output, num_workers=args.num_workers)