"""
HAI Deepfake Detection - Training Script

학습 코드 엔트리 포인트
"""

import argparse
import os
import shutil
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

    # 모델 초기화
    print(f"Initializing model: {config['model']['name']}")
    processor = ViTImageProcessor.from_pretrained(config['model']['name'])
    model = DeepfakeDetector(
        model_name=config['model']['name'],
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained']
    ).to(device)

    # 데이터셋 준비
    train_dir = config['data']['train_dir']
    val_dir = config['data'].get('val_dir', None)
    print(f"Loading training data from: {train_dir}")
    
    # 디버그 모드일 때 설정 조정
    if args.debug:
        config['training']['epochs'] = 2  # 빠르게 2에포크만
        config['training']['batch_size'] = 2
        print("Debug mode enabled: epochs=2, batch_size=2")

    train_dataset = DeepfakeDataset(
        data_dir=train_dir,
        processor=processor,
        num_frames=config['data']['num_frames']
    )
    
    # 데이터가 없으면 경고
    if len(train_dataset) == 0:
        print("Error: No training data found! Please run 'create_dummy_data.py' first.")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0 if args.debug else config['training']['num_workers']
    )
    
    print(f"Training samples: {len(train_dataset)}")

    # 검증 데이터 로더 설정
    val_loader = None
    if val_dir and os.path.exists(val_dir):
        print(f"Loading validation data from: {val_dir}")
        val_dataset = DeepfakeDataset(
            data_dir=val_dir,
            processor=processor,
            num_frames=config['data']['num_frames']
        )
        if len(val_dataset) > 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=config['validation']['batch_size'],
                shuffle=False,
                num_workers=config['validation'].get('num_workers', 2)
            )
            print(f"Validation samples: {len(val_dataset)}")
    
    if not val_loader:
        print("Warning: Validation skipped (no validation data found)")

    # 손실 함수 및 옵티마이저
    class_weights = torch.tensor([1.0, 3.0]).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config['training']['learning_rate']),
        weight_decay=float(config['training']['weight_decay'])
    )

    # Mixed Precision
    scaler = torch.cuda.amp.GradScaler() if config['training']['mixed_precision'] and device == 'cuda' else None

    # 체크포인트에서 재개
    start_epoch = 0
    best_auc = 0.0

    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer, device)
        start_epoch = checkpoint['epoch'] + 1
        best_auc = checkpoint.get('val_auc', 0.0)

    # 학습 루프
    print("\n=== Start Training ===")
    
    for epoch in range(start_epoch, config['training']['epochs']):
        print(f"\n=== Epoch {epoch + 1}/{config['training']['epochs']} ===")

        # 학습
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        print(f"Train Loss: {train_loss:.4f}")

        # 검증
        val_loss, val_auc = 0.0, 0.0
        if val_loader:
            val_loss, val_auc = validate(model, val_loader, criterion, device)
            print(f"Validation - Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")
        
        # 체크포인트 저장
        is_best = val_auc > best_auc
        if is_best:
            best_auc = val_auc

        ckpt_dir = config['experiment'].get('output_dir', 'checkpoints')
        save_checkpoint(
            model, optimizer, epoch, val_auc=val_auc,
            checkpoint_dir=ckpt_dir,
            is_best=is_best
        )

        # Google Drive 백업 (Colab 환경인 경우)
        drive_ckpt_dir = '/content/drive/MyDrive/HAI_Deepfake/checkpoints'
        if os.path.exists('/content/drive'):
            try:
                os.makedirs(drive_ckpt_dir, exist_ok=True)
                
                # 현재 에포크 체크포인트 복사
                ckpt_filename = f'checkpoint_epoch_{epoch:03d}.pt'
                src_ckpt = os.path.join(ckpt_dir, ckpt_filename)
                if os.path.exists(src_ckpt):
                    shutil.copy2(src_ckpt, os.path.join(drive_ckpt_dir, ckpt_filename))
                    print(f"Backed up checkpoint to Drive: {ckpt_filename}")
                
                # 최고 성능 모델 복사
                if is_best:
                    src_best = os.path.join(ckpt_dir, 'best_model.pt')
                    if os.path.exists(src_best):
                        shutil.copy2(src_best, os.path.join(drive_ckpt_dir, 'best_model.pt'))
                        print("Backed up best_model.pt to Drive")
            except Exception as e:
                print(f"Warning: Failed to backup to Google Drive: {e}")

    print("\nTraining completed successfully!")


if __name__ == '__main__':
    main()
