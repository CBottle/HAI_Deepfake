"""
유틸리티 함수 모듈

공통으로 사용되는 헬퍼 함수들을 정의합니다.
"""

import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42):
    """
    재현성을 위한 랜덤 시드 설정

    Args:
        seed: 랜덤 시드 값
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # CuDNN 설정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """
    YAML 설정 파일 로드

    Args:
        config_path: 설정 파일 경로

    Returns:
        설정 딕셔너리
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_auc: float,
    checkpoint_dir: str = 'checkpoints',
    is_best: bool = False
):
    """
    모델 체크포인트 저장

    Args:
        model: 저장할 모델
        optimizer: 옵티마이저
        epoch: 현재 에포크
        val_auc: 검증 AUC
        checkpoint_dir: 체크포인트 저장 디렉토리
        is_best: 최고 성능 모델 여부
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_auc': val_auc
    }

    # Epoch별 체크포인트
    checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch:03d}.pt'
    torch.save(checkpoint, checkpoint_path)
    print(f'Checkpoint saved: {checkpoint_path}')

    # 최고 성능 모델 저장
    if is_best:
        best_path = checkpoint_dir / 'best_model.pt'
        torch.save(checkpoint, best_path)
        print(f'Best model updated! AUC: {val_auc:.4f}')


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = 'cuda'
) -> Dict:
    """
    체크포인트 로드

    Args:
        checkpoint_path: 체크포인트 파일 경로
        model: 모델
        optimizer: 옵티마이저 (Optional)
        device: 디바이스

    Returns:
        체크포인트 딕셔너리
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Validation AUC: {checkpoint['val_auc']:.4f}")

    return checkpoint


class AverageMeter:
    """
    평균 및 현재 값을 추적하는 미터
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def cleanup_checkpoints(checkpoint_dir: str, keep_last: int = 5):
    """
    오래된 체크포인트 삭제 (최근 N개만 유지)

    Args:
        checkpoint_dir: 체크포인트 디렉토리
        keep_last: 유지할 체크포인트 개수
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = sorted(checkpoint_dir.glob('checkpoint_epoch_*.pt'))

    if len(checkpoints) > keep_last:
        for old_ckpt in checkpoints[:-keep_last]:
            old_ckpt.unlink()
            print(f'Removed old checkpoint: {old_ckpt}')


def get_device() -> str:
    """
    사용 가능한 디바이스 반환

    Returns:
        'cuda' 또는 'cpu'
    """
    return 'cuda' if torch.cuda.is_available() else 'cpu'
