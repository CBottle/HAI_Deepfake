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
    parser.add_argument('--unfreeze', action='store_true',
                        help='Stage 2: Unfreeze backbone for full fine-tuning')
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
    for i, batch in enumerate(pbar):
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        # Mixed Precision Training
        if scaler is not None: # (현재 코드엔 scaler가 없지만 구조 유지)
            pass
        
        logits = model(pixel_values)
        loss = criterion(logits, labels)
        
        # [긴급 디버깅] 첫 번째 배치의 상태 확인
        if i == 0:
            print(f"\n[Debug] Logits Range: Min={logits.min().item():.4f}, Max={logits.max().item():.4f}")
            print(f"[Debug] Labels Sample: {labels[:10].cpu().numpy()}")
            
            # 라벨 반전 테스트
            loss_inverted = criterion(logits, 1 - labels)
            print(f"[Debug] Original Loss: {loss.item():.4f} vs Inverted Label Loss: {loss_inverted.item():.4f}")
            
            if loss_inverted.item() < loss.item():
                print("🚨 [WARNING] 라벨이 반대일 확률이 매우 높습니다! (Inverted Loss가 더 낮음)")

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
    from sklearn.model_selection import GroupShuffleSplit # 그룹 스플릿 추가

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
        balanced_df = pd.concat([s_real, s_fake]).reset_index(drop=True)
        
        # [Data Leakage 방지] 비디오 ID 추출 및 그룹 스플릿
        # 파일명 예시: 'video_01_frame0.jpg', 'aomwayen.mp4_frame10.jpg'
        # 전략: 뒤에서 첫 번째 '_' 기준 앞부분을 비디오 ID로 간주
        balanced_df['video_id'] = balanced_df['filename'].apply(lambda x: x.rsplit('_', 1)[0] if '_' in x else x)
        
        gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
        train_idx, val_idx = next(gss.split(balanced_df, groups=balanced_df['video_id']))
        
        train_df = balanced_df.iloc[train_idx]
        val_df = balanced_df.iloc[val_idx]
        
        print(f"📊 [그룹 스플릿] 데이터 준비 완료: 총 {len(balanced_df)}장")
        print(f"   - 학습(Train): {len(train_df)}장 (Videos: {train_df['video_id'].nunique()})")
        print(f"   - 검증(Val):   {len(val_df)}장 (Videos: {val_df['video_id'].nunique()})")
        
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

    # [Training Stage Selection]
    if args.unfreeze:
        # [Stage 2: Full Fine-tuning with Differential LR]
        print("🔓 [Stage 2] Unfreezing All Layers with Differential LR...")
        for param in model.parameters():
            param.requires_grad = True
            
        # 파라미터 그룹 분리 (Backbone vs Head)
        backbone_params = []
        head_params = []
        
        # Head 이름 찾기 (timm 호환)
        head_name = 'classifier' if hasattr(model.model, 'classifier') else 'fc' if hasattr(model.model, 'fc') else 'head'
        
        for name, param in model.named_parameters():
            if head_name in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
        
        # 차등 학습률 적용
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': 1e-6}, # 몸통: 지식 보존 (아주 살살)
            {'params': head_params, 'lr': 1e-4}      # 머리: 빠른 적응
        ], weight_decay=0.01)
    else:
        # [Stage 1: SRM Warmup]
        print("🔒 [Stage 1] Freezing Backbone Body for SRM Adaptation...")
        
        # 1. 전체 백본 Freeze
        for param in model.model.parameters():
            param.requires_grad = False
            
        # 2. 첫 번째 레이어 (conv_stem) Unfreeze
        for param in model.model.conv_stem.parameters():
            param.requires_grad = True
            
        # 3. 분류기 (classifier) Unfreeze
        head = getattr(model.model, 'classifier', getattr(model.model, 'fc', getattr(model.model, 'head', None)))
        if head:
            for param in head.parameters():
                param.requires_grad = True

        # 학습할 파라미터만 골라서 Optimizer에 전달 (높은 LR)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(trainable_params, lr=1e-3, weight_decay=0.01)
    
    # 스케줄러: 웜업(Warmup) 후 코사인 어닐링
    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
    
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=len(train_loader))
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=config['training']['epochs'] * len(train_loader))
    
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
