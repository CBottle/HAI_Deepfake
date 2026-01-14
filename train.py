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
    args = parse_args()
    config = load_config(args.config)
    set_seed(config['experiment']['seed'])

    # 1. CPU 강제 설정 (테스트용)
    device = torch.device('cpu') 
    print(f"Device forced to: {device}")

    # 2. 데이터 100장 샘플링 및 임시 CSV 생성
    import pandas as pd
    full_df = pd.read_csv(config['data']['train_csv'])
    
    # Label이 0(Real), 1(Fake)라고 가정 (데이터에 맞춰 확인해!)
    df_real = full_df[full_df['label'] == 0].sample(n=min(50, len(full_df[full_df['label']==0])), random_state=42)
    df_fake = full_df[full_df['label'] == 1].sample(n=min(50, len(full_df[full_df['label']==1])), random_state=42)
    tiny_df = pd.concat([df_real, df_fake]).reset_index(drop=True)
    
    # 임시 CSV 저장 (DeepfakeDataset이 경로를 받으므로)
    tiny_csv_path = 'config/tiny_train.csv'
    tiny_df.to_csv(tiny_csv_path, index=False)
    print(f"✅ Tiny Dataset 생성 완료 (100장): {tiny_csv_path}")

    # 모델 초기화
    processor = ViTImageProcessor.from_pretrained(config['model']['name'])
    model = DeepfakeDetector(
        model_name=config['model']['name'],
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained']
    ).to(device)

    # 3. 샌니티 체크를 위해 모든 레이어 열기 (Unfreeze)
    # 3에포크 기다리지 말고 지금 바로 다 학습 가능하게 만들어
    for param in model.parameters():
        param.requires_grad = True
    print("🚀 All layers unfrozen for Sanity Check.")

    # 1. 전처리 규칙 정의 (Resize + Normalize)
    val_transform = A.Compose([
        A.Resize(224, 224), # ViT 기본 입력 크기
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]) 
    
    # 데이터셋 준비 (샘플링한 CSV 경로 사용)
    train_dataset = DeepfakeDataset(
        csv_path=tiny_csv_path, # 임시 CSV 사용
        img_dir=config['data']['img_dir'],
        processor=processor,
        num_frames=config['data']['num_frames'],
        transform=val_transform # 샌니티 체크는 증강 없이 깔끔하게 테스트!
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=4, # CPU니까 배치는 작게
        shuffle=True
    )

    # 옵티마이저 (학습 반응을 보기 위해 LR을 조금 높게 설정)
    optimizer = torch.optim.AdamW([
        {'params': model.model.vit.parameters(), 'lr': 1e-4}, 
        {'params': model.model.classifier.parameters(), 'lr': 1e-3}
    ], weight_decay=0.05)

    # ⚠️ 네 코드에 scheduler가 주석처리 되어있어서 에러 날 수 있어!
    # 테스트할 때는 아래 한 줄을 활성화하거나, 루프 안의 scheduler.step()을 주석처리해.
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 학습 루프 (샌니티 체크는 10~20 에포크만 봐도 충분해)
    print("\n=== Start Sanity Check (100 Samples) ===")
    for epoch in range(20):
        train_loss = train_epoch(model, train_loader, torch.nn.CrossEntropyLoss(), optimizer, device)
        
        # 100장에 대한 AUC 직접 계산해서 출력해보기
        # (validate 함수를 tiny_loader에 대해 돌려도 돼)
        _, tiny_auc = validate(model, train_loader, torch.nn.CrossEntropyLoss(), device)
        
        print(f"Epoch {epoch+1} - Loss: {train_loss:.4f}, AUC: {tiny_auc:.4f}")
        
        if tiny_auc > 0.95:
            print("🎉 Success! 모델이 100장의 데이터를 학습하기 시작했어.")
            break

if __name__ == '__main__':
    main()
