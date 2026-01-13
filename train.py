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
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

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

    # 처음에는 ViT의 몸통(Backbone)은 얼리고 분류기(Head)만 학습하자
    for param in model.parameters():
        param.requires_grad = False

    # 2. 마지막 분류기(classifier)만 다시 녹여서 공부하게 만들어
    # 모델 내부에 'classifier'라는 이름이 들어간 레이어만 찾아서requires_grad를 True로 바꿔줘
    for name, param in model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True
            print(f"Layer {name} is unfrozen and ready to learn!")

    print("ViT Backbone is frozen. Only the classifier will be trained for the first 3 epochs.")

    # 데이터셋 준비
    train_dir = config['data']['train_dir']
    val_dir = config['data'].get('val_dir', None)
    print(f"Loading training data from: {train_dir}")
    
    # 디버그 모드일 때 설정 조정
    if args.debug:
        config['training']['epochs'] = 2  # 빠르게 2에포크만
        config['training']['batch_size'] = 2
        print("Debug mode enabled: epochs=2, batch_size=2")

    hard_transform = A.Compose([
        # 1. 기하학적 변형 (얼굴 각도와 구도를 계속 바꿈)
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.7),
    
        # 2. 강한 노이즈 & 화질 저하 (딥페이크는 압축 노이즈에 약해!)
        A.OneOf([
            A.ImageCompression(quality_lower=30, quality_upper=70, p=0.5), # 화질 확 깨기
            A.GaussNoise(var_limit=(20.0, 100.0), p=0.5), # 지지직거리는 노이즈
            A.ISONoise(p=0.5),
        ], p=0.6),

        # 3. 색감 & 조명 테러 (피부 톤이나 조명에 의존하지 못하게)
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    
        # 4. 필살기: CoarseDropout (Cutout)
        # 얼굴의 일부분을 검은 사각형으로 가려버려. 
        # 눈 하나가 없어도 다른 부분(입가, 턱선)의 조작 흔적을 찾게 만드는 훈련이야!
        A.CoarseDropout(
            max_holes=8, 
            max_height=32, 
            max_width=32, 
            min_holes=2, 
            p=0.5
        ),
    
    # 5. 공간적 왜곡
    A.GridDistortion(p=0.3), 
    ])

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
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW([
    # ViT 백본: 아주 조심스럽게 (기존 LR의 1/100 수준)
    {'params': model.vit.parameters(), 'lr': 1e-6}, 
    # 분류기(Head): 원래 속도로
    {'params': model.classifier.parameters(), 'lr': 1e-4}
    ], weight_decay=0.05) # Weight Decay를 좀 더 높여서 암기를 방지해!  

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

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

        # 예: 3에포크 이후부터는 몸통도 같이 학습 (Fine-tuning)
        if epoch == 3:
            for param in model.model.parameters():
                param.requires_grad = True
            print("ViT Backbone unfrozen. Fine-tuning the whole model...")

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

        # 에포크 마지막에 스케줄러 업데이트
        scheduler.step()
        print(f"Current LR: {scheduler.get_last_lr()[0]}")

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
