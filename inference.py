"""
HAI Deepfake Detection - Inference Script

추론 코드 엔트리 포인트
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import timm  # timm 라이브러리 추가
from timm.data import resolve_data_config, create_transform # timm 전처리 도구
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import InferenceDataset
from src.utils import set_seed, load_config, get_device


def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description='Inference Deepfake Detection Model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model checkpoint (optional)')
    parser.add_argument('--test_dir', type=str, default=None,
                        help='Path to test data directory')
    parser.add_argument('--output', type=str, default='submissions/submission.csv',
                        help='Path to output submission file')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for inference')
    return parser.parse_args()


def infer_batch(model, pixel_values, device):
    """
    배치 추론

    Args:
        model: 추론할 모델
        pixel_values: 입력 이미지 텐서
        device: 디바이스

    Returns:
        Fake 확률 리스트
    """
    with torch.no_grad():
        pixel_values = pixel_values.to(device)
        # 모델 출력 처리 (Tensor vs ImageClassifierOutput)
        outputs = model(pixel_values)
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs
            
        probs = F.softmax(logits, dim=1)[:, 1]  # Fake 확률
        return probs.cpu().tolist()


def main():
    """메인 추론 루프"""
    args = parse_args()

    # 설정 로드
    config = load_config(args.config)

    # 시드 설정
    set_seed(config['experiment']['seed'])

    # 디바이스 설정
    device = get_device()
    print(f"Device: {device}")

    # 경로 설정
    test_dir = args.test_dir or config['data']['test_dir']
    test_dir = Path(test_dir)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    batch_size = args.batch_size or config['inference']['batch_size']

    # 모델 로드
    print("Loading model...")
    model_name = config['model']['name']

    if args.model:
        # 학습된 체크포인트에서 로드
        from src.models import load_model
        model = load_model(args.model, model_name, device)
        print(f"Loaded checkpoint: {args.model}")
    else:
        # 사전학습 모델 직접 로드 (timm)
        model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=config['model']['num_classes']
        ).to(device)
        print(f"Loaded pretrained model (timm): {model_name}")

    model.eval()
    
    # [중요] 라벨 맵 확인: 0이 무엇이고 1이 무엇인지 출력 (0.9 점프의 핵심 힌트)
    if hasattr(model, 'config') and hasattr(model.config, 'id2label'):
        print(f"Model Label Mapping (id2label): {model.config.id2label}")
    elif hasattr(model, 'pretrained_cfg'):
        print(f"Model Label Mapping (pretrained_cfg): {model.pretrained_cfg.get('label_names', 'Not Found')}")
    else:
        print("Model Label Mapping: Not explicitly found in config. Assuming [Real, Fake].")

    # timm 모델에 최적화된 Transform 생성
    # 모델의 학습 설정(data_config)을 읽어와서 자동으로 Resize, Normalize 등을 설정함
    data_config = resolve_data_config(model.default_cfg, model=model)
    transform = create_transform(**data_config)
    print(f"Data Config: {data_config}")

    # 데이터셋 준비
    print(f"Loading test data from: {test_dir}")
    # 추론 시에는 비디오 프레임을 여러 장(30장) 봐여 정확도가 오름
    # 1시간 제한 내에서 최대한 정밀하게 검사하기 위해 30장으로 설정
    dataset = InferenceDataset(
        data_dir=str(test_dir),
        transform=transform,
        num_frames=30
    )

    print(f"Test data length: {len(dataset)}")
    
    # DataLoader를 이용한 배치 추론 (속도와 메모리 균형)
    # Colab T4 GPU(16GB VRAM) 및 384x384 해상도 기준 batch_size=8 권장
    inference_batch_size = 8
    dataloader = DataLoader(
        dataset, 
        batch_size=inference_batch_size, 
        shuffle=False, 
        num_workers=0, # 메모리 부족 및 NumPy 충돌 방지를 위해 0으로 설정
        pin_memory=True
    )

    # 추론
    results = {}
    print(f"Running inference (Batch Size: {inference_batch_size})...")
    
    with torch.no_grad():
        for pixel_values, filenames, _ in tqdm(dataloader, desc="Processing"):
            # pixel_values: (B, T, C, H, W) -> (B*T, C, H, W)로 펼침
            b, t, c, h, w = pixel_values.shape
            pixel_values = pixel_values.view(-1, c, h, w).to(device)
            
            # 모델 출력 처리
            outputs = model(pixel_values)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            # 확률 계산 (B*T, 2)
            probs = F.softmax(logits, dim=1)[:, 1]
            
            # 다시 (B, T)로 묶어서 영상별 평균 계산
            probs = probs.view(b, t)
            avg_probs = probs.mean(dim=1).cpu().numpy()
            
            for filename, prob in zip(filenames, avg_probs):
                results[filename] = float(prob)

    print(f"Inference completed. Processed: {len(results)} files")

    # 제출 파일 생성
    print(f"Creating submission file: {output_path}")

    # sample_submission.csv 읽기 (있는 경우)
    sample_path = Path('sample_submission.csv')
    if sample_path.exists():
        submission = pd.read_csv(sample_path)
        submission['prob'] = submission['filename'].map(results).fillna(0.0)
    else:
        # sample_submission.csv가 없으면 직접 생성
        submission = pd.DataFrame({
            'filename': list(results.keys()),
            'prob': list(results.values())
        })

    # CSV 저장
    submission.to_csv(output_path, encoding='utf-8-sig', index=False)
    print(f"Submission saved to: {output_path}")
    print(f"\nSample predictions:")
    print(submission.head(10))


if __name__ == '__main__':
    main()
