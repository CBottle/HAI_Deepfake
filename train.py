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


# 학습 데이터를 위한 '순한 맛' 증강 설정 (Soft Augmentation)
# 화질을 손상시키지 않고 형태의 다양성만 확보합니다.
soft_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.2),
])


def main():
    """순한 맛 증강 버전 - 고득점 Fine-tuning용"""
    args = parse_args()
    config = load_config(args.config)
    set_seed(config['experiment']['seed'])

    # 1. 디바이스 설정
    device = get_device() 
    print(f"🚀 학습 시작! 사용 디바이스: {device}")

    # 모델 초기화
    # Processor는 Hugging Face의 ViT용을 빌려 씀 (EfficientNet도 224x224 Normalize라 호환됨)
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = DeepfakeDetector(
        model_name=config['model']['name'],
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained']
    ).to(device)

    # 2. 데이터 샘플링 (Real:Fake = 1:1 밸런싱)
    import pandas as pd
    from sklearn.model_selection import train_test_split

    train_csv_path = config['data']['train_csv']
    if os.path.exists(train_csv_path):
        full_df = pd.read_csv(train_csv_path)
        
        # 클래스 분리
        df_real = full_df[full_df['label'] == 0]
        df_fake = full_df[full_df['label'] == 1]
        
        target_per_class = 15000
        
        # 각 클래스에서 1.5만 장씩 샘플링
        s_real = df_real.sample(n=min(target_per_class, len(df_real)), random_state=42)
        s_fake = df_fake.sample(n=min(target_per_class, len(df_fake)), random_state=42)
        
        # 데이터 병합
        balanced_df = pd.concat([s_real, s_fake]).sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Train / Val 분리 (9:1)
        train_df, val_df = train_test_split(balanced_df, test_size=0.1, random_state=42, stratify=balanced_df['label'])
        
        print(f"📊 [순한맛] 데이터 준비 완료: 총 {len(balanced_df)}장")
        print(f"   - 학습(Train): {len(train_df)}장")
        print(f"   - 검증(Val):   {len(val_df)}장")
        
        # 임시 파일 저장
        train_df.to_csv("temp_train.csv", index=False)
        val_df.to_csv("temp_val.csv", index=False)
    else:
        raise FileNotFoundError(f"⚠️ CSV 파일을 찾을 수 없습니다: {train_csv_path}")

    # 3. 데이터셋 및 로더
    train_dataset = DeepfakeDataset(
        csv_path="temp_train.csv",
        img_dir=config['data']['img_dir'],
        processor=processor,
        num_frames=config['data']['num_frames'],
        transform=soft_transform # 순한맛 적용
    )
    
    val_dataset = DeepfakeDataset(
        csv_path="temp_val.csv",
        img_dir=config['data']['img_dir'],
        processor=processor,
        num_frames=config['data']['num_frames'],
        transform=None 
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True if device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True if device == 'cuda' else False
    )

    # 4. 옵티마이저 (초반 수렴 속도를 위해 LR 상향: 1e-5 -> 1e-4)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # 스케줄러: 웜업(Warmup) 후 코사인 어닐링
    # 초반 1에포크 동안은 학습률을 서서히 올리고, 그 뒤로는 서서히 낮춤 (안정적 학습)
    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
    
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=len(train_loader))
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=config['training']['epochs'] * len(train_loader))
    
    # 1에포크 웜업 후 나머지 기간 코사인 어닐링
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[len(train_loader)])
    criterion = torch.nn.CrossEntropyLoss()

    # 체크포인트 로드 (Resume)
    start_epoch = 0
    best_auc = 0.0
    
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"🔄 Resuming from checkpoint: {args.resume}")
            checkpoint = load_checkpoint(args.resume, model, optimizer, device)
            start_epoch = checkpoint['epoch'] + 1
            if 'val_auc' in checkpoint:
                best_auc = checkpoint['val_auc']
            print(f"   -> Resuming from Epoch {start_epoch+1}")
        else:
            print(f"⚠️ Checkpoint not found: {args.resume}")

    # 학습 루프
    print(f"\n=== Start Fine-tuning (Total Epochs: {config['training']['epochs']}) ===")
    ckpt_dir = config['training']['experiment']['output_dir']
    
    for epoch in range(start_epoch, config['training']['epochs']):
        # 1. 학습
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 2. 검증
        val_loss, val_auc = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{config['training']['epochs']} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")
        
        # 3. 체크포인트 저장
        is_best = val_auc > best_auc
        if is_best:
            best_auc = val_auc
            print(f"🏆 Best AUC Updated: {best_auc:.4f}")
        
        save_checkpoint(
            model, 
            optimizer, 
            epoch, 
            val_auc, 
            checkpoint_dir=ckpt_dir, 
            is_best=is_best
        )

        scheduler.step()

if __name__ == '__main__':
    main()
