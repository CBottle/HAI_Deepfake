"""
모델 정의 모듈

Dual-Stream Network (RGB + SRM) 구조로 딥페이크 탐지를 수행합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np


class SRMConv2d(nn.Module):
    """
    SRM (Spatial Rich Model) 필터 레이어
    이미지의 텍스처 및 노이즈 정보를 추출하기 위해 고정된 3개의 커널을 사용합니다.
    """
    def __init__(self, inc=3):
        super().__init__()
        self.inc = inc
        self.truc = nn.Hardtanh(-3, 3)
        
        # SRM 필터 커널 정의 (5x5)
        # 1. Spam 14h (수평/수직 엣지)
        # 2. Spam 14v
        # 3. MinMax
        
        q = [4.0, 12.0, 2.0]
        filter1 = [[0, 0, 0, 0, 0],
                   [0, -1, 2, -1, 0],
                   [0, 2, -4, 2, 0],
                   [0, -1, 2, -1, 0],
                   [0, 0, 0, 0, 0]]
        filter2 = [[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]]
        filter3 = [[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 1, -2, 1, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]]
        
        filter1 = np.array(filter1, dtype=float) / q[0]
        filter2 = np.array(filter2, dtype=float) / q[1]
        filter3 = np.array(filter3, dtype=float) / q[2]
        
        filters = np.array([[filter1, filter1, filter1], 
                            [filter2, filter2, filter2], 
                            [filter3, filter3, filter3]])  # (3, 3, 5, 5)
        
        self.conv = nn.Conv2d(inc, 3, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv.weight.data = torch.tensor(filters, dtype=torch.float32)
        
        # 학습되지 않도록 고정 (Freeze)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.conv(x)
        out = self.truc(out)
        return out


class DualStreamDetector(nn.Module):
    """
    RGB Stream + SRM Stream 듀얼 구조 모델
    """
    def __init__(
        self,
        model_name: str = "tf_efficientnetv2_m.in21k",
        num_classes: int = 2,
        pretrained: bool = True
    ):
        super().__init__()

        # Stream 1: RGB (기존 모델)
        # num_classes=0으로 설정하여 Classification Head 없이 Feature만 뽑음
        self.rgb_stream = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0 
        )
        
        # Stream 2: SRM (노이즈 분석용 가벼운 모델)
        self.srm_layer = SRMConv2d()
        self.srm_stream = timm.create_model(
            'efficientnet_b0', # 가볍고 빠른 모델 사용
            pretrained=True,
            num_classes=0,
            in_chans=3 # SRM 출력(3채널)을 받음
        )
        
        # Feature Dimension 계산
        # EfficientNetV2-M: 1280, EfficientNet-B0: 1280 -> Total 2560
        rgb_dim = self.rgb_stream.num_features
        srm_dim = self.srm_stream.num_features
        concat_dim = rgb_dim + srm_dim
        
        # Fusion Head (MLP)
        self.classifier = nn.Sequential(
            nn.Linear(concat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Stream 1: RGB
        rgb_feat = self.rgb_stream(x) # (Batch, 1280)
        
        # Stream 2: SRM
        srm_x = self.srm_layer(x)     # (Batch, 3, H, W)
        srm_feat = self.srm_stream(srm_x) # (Batch, 1280)
        
        # Fusion
        combined = torch.cat([rgb_feat, srm_feat], dim=1) # (Batch, 2560)
        logits = self.classifier(combined)
        
        return logits


# 기존 호환성을 위해 클래스명 유지 (내부적으로 DualStream 사용)
DeepfakeDetector = DualStreamDetector


def load_model(checkpoint_path: str, model_name: str = "tf_efficientnetv2_m.in21k", device: str = "cuda") -> DualStreamDetector:
    """
    체크포인트에서 DualStreamDetector 모델 로드
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = DualStreamDetector(model_name=model_name)
    
    # 가중치 키 매핑 (혹시 모를 불일치 대비)
    state_dict = checkpoint['model_state_dict']
    
    # 만약 기존 단일 모델 체크포인트를 로드하려고 한다면? (rgb_stream에만 넣어야 함)
    # -> 이 경우는 'Fine-tuning'이므로 별도 처리가 필요하지만, 
    #    여기서는 'DualStream'으로 학습된 체크포인트를 로드한다고 가정.
    
    model.load_state_dict(state_dict, strict=False) # strict=False로 유연하게 로드
    model.to(device)
    model.eval()

    return model
