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


# DFDC 얼굴 크롭 데이터셋에 적합한 강력한 증강 설정
hard_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    # 압축 손실: 딥페이크 탐지 모델이 저화질/압축된 환경에서도 잘 작동하게 함
    A.ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
    # 블러/노이즈: 다양한 캡처 환경 시뮬레이션
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 7)),
        A.GaussNoise(var_limit=(10.0, 50.0)),
    ], p=0.3),
    # 밝기/대비 및 기하학적 변환
    A.RandomBrightnessContrast(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.3),
])


def main():
    """3만 장 샘플링 및 GPU 학습 버전"""
    args = parse_args()
    config = load_config(args.config)
    set_seed(config['experiment']['seed'])

    # 1. 디바이스 설정
    device = get_device() 
    print(f"🚀 학습 시작! 사용 디바이스: {device}")

    # 모델 초기화
    processor = ViTImageProcessor.from_pretrained(config['model']['name'])
    model = DeepfakeDetector(
        model_name=config['model']['name'],
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained']
    ).to(device)

    # 2. 데이터 샘플링 (3만 장) - 사용자 요청에 따라 밸런싱 미수행
    import pandas as pd
    train_csv_path = config['data']['train_csv']
    if os.path.exists(train_csv_path):
        full_df = pd.read_csv(train_csv_path)
        target_samples = 30000
        
        if len(full_df) > target_samples:
            train_df = full_df.sample(n=target_samples, random_state=42).reset_index(drop=True)
            print(f"📊 {len(full_df)}장 중 {target_samples}장 랜덤 샘플링 완료 (밸런싱 미수행)")
        else:
            train_df = full_df
            print(f"📊 전체 데이터({len(full_df)}장)를 사용합니다.")
        
        # Dataset 클래스가 csv_path만 받으므로 임시 파일 저장
        temp_train_csv = "temp_train_sampled.csv"
        train_df.to_csv(temp_train_csv, index=False)
        current_train_csv = temp_train_csv
    else:
        raise FileNotFoundError(f"⚠️ CSV 파일을 찾을 수 없습니다: {train_csv_path}")

    # 3. 데이터셋 및 로더
    train_dataset = DeepfakeDataset(
        csv_path=current_train_csv,
        img_dir=config['data']['img_dir'],
        processor=processor,
        num_frames=config['data']['num_frames'],
        transform=hard_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True if device == 'cuda' else False
    )

    # 4. 옵티마이저 및 스케줄러 (3만 장에 맞게 T_max 조절)
    optimizer = optim.AdamW([
        {'params': model.model.vit.parameters(), 'lr': 1e-5},
        {'params': model.model.classifier.parameters(), 'lr': 5e-4}
    ], weight_decay=0.05)
    
    # 3만 장이면 1에폭에 스텝이 많지 않으니 T_max를 에폭 수에 맞춰
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])


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
