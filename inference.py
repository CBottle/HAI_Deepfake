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
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTImageProcessor
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
        # 사전학습 모델 직접 로드
        model = ViTForImageClassification.from_pretrained(model_name).to(device)
        print(f"Loaded pretrained model: {model_name}")

    model.eval()

    # 프로세서 로드
    # timm 모델을 쓰더라도 전처리는 ViT 표준(224x224, ImageNet Mean)을 따르므로 호환됨
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

    # 데이터셋 준비
    print(f"Loading test data from: {test_dir}")
    dataset = InferenceDataset(
        data_dir=str(test_dir),
        processor=processor,
        num_frames=config['data']['num_frames']
    )

    print(f"Test data length: {len(dataset)}")

    # 추론
    results = {}
    debug_dir = Path('submissions/debug_frames')
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_count = 0

    print("Running inference...")
    for idx in tqdm(range(len(dataset)), desc="Processing"):
        pixel_values, filename, processed_frames = dataset[idx] # processed_frames는 PIL 이미지 리스트

        # 디버깅용 이미지 저장 (최초 5개 샘플의 프레임들)
        if debug_count < 5:
            for i, frame in enumerate(processed_frames):
                frame.save(debug_dir / f"debug_{filename}_f{i}.png")
            debug_count += 1

        if len(processed_frames) > 0:
            # 프레임별 추론
            probs = infer_batch(model, pixel_values, device)
            # 프레임 평균
            final_prob = float(np.mean(probs))
        else:
            # 에러 시 0.0 (Real)
            final_prob = 0.0

        results[filename] = final_prob

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
