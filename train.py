"""
HAI Deepfake Detection - Training Script

학습 코드 엔트리 포인트
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import ViTImageProcessor
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from src.models import DeepfakeDetector
from src.dataset import DeepfakeDataset
from src.utils import (
    set_seed,
    load_config,
    save_checkpoint,
    load_checkpoint,
    get_device,
    AverageMeter
)


def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description='Train Deepfake Detection Model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode (small dataset)')
    return parser.parse_args()


def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """
    한 에포크 학습

    Args:
        model: 학습할 모델
        dataloader: 데이터 로더
        criterion: 손실 함수
        optimizer: 옵티마이저
        device: 디바이스
        scaler: GradScaler (Mixed Precision)

    Returns:
        평균 손실
    """
    model.train()
    loss_meter = AverageMeter()

    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        # Mixed Precision Training
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(pixel_values)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(pixel_values)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        loss_meter.update(loss.item(), pixel_values.size(0))
        pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})

    return loss_meter.avg


def validate(model, dataloader, criterion, device):
    """
    검증

    Args:
        model: 모델
        dataloader: 데이터 로더
        criterion: 손실 함수
        device: 디바이스

    Returns:
        평균 손실, ROC-AUC
    """
    model.eval()
    loss_meter = AverageMeter()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for batch in pbar:
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)

            logits = model(pixel_values)
            loss = criterion(logits, labels)

            probs = torch.softmax(logits, dim=1)[:, 1]  # Fake 확률

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            loss_meter.update(loss.item(), pixel_values.size(0))
            pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})

    # ROC-AUC 계산
    auc = roc_auc_score(all_labels, all_probs)

    return loss_meter.avg, auc


def main():
    """메인 학습 루프"""
    args = parse_args()

    # 설정 로드
    config = load_config(args.config)

    # 시드 설정
    set_seed(config['experiment']['seed'])

    # 디바이스 설정
    device = get_device() if not args.debug else 'cpu'
    print(f"Device: {device}")

    # 데이터셋 준비 (실제 구현 시 레이블이 있는 데이터셋 필요)
    # TODO: 실제 학습 데이터로 교체
    print("Note: 현재는 추론만 가능한 baseline입니다.")
    print("학습을 위해서는 레이블이 있는 데이터셋이 필요합니다.")
    print("데이터셋 준비 후 dataset.py의 DeepfakeDataset을 사용하세요.")

    # 모델 초기화
    processor = ViTImageProcessor.from_pretrained(config['model']['name'])
    model = DeepfakeDetector(
        model_name=config['model']['name'],
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained']
    ).to(device)

    # 손실 함수 및 옵티마이저
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Mixed Precision
    scaler = torch.cuda.amp.GradScaler() if config['training']['mixed_precision'] else None

    # 체크포인트에서 재개
    start_epoch = 0
    best_auc = 0.0

    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer, device)
        start_epoch = checkpoint['epoch'] + 1
        best_auc = checkpoint.get('val_auc', 0.0)

    # 학습 루프 예시
    # TODO: 실제 학습 데이터로 교체 필요
    print("\n=== Training Example (Placeholder) ===")
    print("실제 학습을 위해서는:")
    print("1. 학습 데이터 준비 (train_data/, val_data/)")
    print("2. 레이블 정보 추가")
    print("3. DataLoader 구성")
    print("4. train_epoch 및 validate 함수 실행")

    # 예시 코드 (실제 사용 시 주석 해제)
    # for epoch in range(start_epoch, config['training']['epochs']):
    #     print(f"\n=== Epoch {epoch + 1}/{config['training']['epochs']} ===")
    #
    #     # 학습
    #     train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
    #     print(f"Train Loss: {train_loss:.4f}")
    #
    #     # 검증
    #     if (epoch + 1) % config['validation']['frequency'] == 0:
    #         val_loss, val_auc = validate(model, val_loader, criterion, device)
    #         print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
    #
    #         # 체크포인트 저장
    #         is_best = val_auc > best_auc
    #         if is_best:
    #             best_auc = val_auc
    #
    #         save_checkpoint(
    #             model, optimizer, epoch, val_auc,
    #             checkpoint_dir='checkpoints',
    #             is_best=is_best
    #         )

    print("\n학습 스크립트 구조 확인 완료!")


if __name__ == '__main__':
    main()
