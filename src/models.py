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


class DeepfakeDetector(nn.Module):
    """
    단일 백본 SRM Early Fusion 모델 (Weight Surgery 적용)
    RGB(3ch) + SRM(3ch) = 6채널 입력을 받는 단일 EfficientNet 모델
    """
    def __init__(
        self,
        model_name: str = "tf_efficientnetv2_m.in21k",
        num_classes: int = 2,
        pretrained: bool = True
    ):
        super().__init__()

        # SRM 필터 레이어
        self.srm_layer = SRMConv2d()
        
        # 정규화 해제(Un-normalize)를 위한 값 설정 (ImageNet 기준)
        self.register_buffer('mean', torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))

        # 1. 단일 백본 생성 (6채널)
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=6
        )

        # 2. 첫 번째 Conv 레이어 가중치 이식 (Weight Surgery)
        if pretrained:
            print(f"💉 Performing Weight Surgery on {model_name} conv_stem...")
            # 순정 3채널 모델에서 가중치 추출
            temp_model = timm.create_model(model_name, pretrained=True, num_classes=0)
            old_weight = temp_model.conv_stem.weight.data # (out_ch, 3, k, k)
            
            # 6채널 모델의 가중치에 이식
            # [0:3] 채널: 기존 RGB 지식 그대로 복사
            self.model.conv_stem.weight.data[:, 0:3, :, :].copy_(old_weight)
            # [3:6] 채널: 기존 지식으로 초기화 (학습 속도 향상)
            self.model.conv_stem.weight.data[:, 3:6, :, :].copy_(old_weight)
            
            del temp_model # 메모리 절약

    def forward(self, x):
        # 1. SRM을 위한 정규화 해제 (SRM은 [0, 1] 또는 [0, 255] 데이터를 선호함)
        # x는 현재 [-1, 1] 또는 정규화된 상태
        with torch.no_grad():
            unnorm_x = x * self.std + self.mean
            unnorm_x = torch.clamp(unnorm_x, 0, 1)
        
        # 2. SRM 특징 추출
        srm_x = self.srm_layer(unnorm_x) # (Batch, 3, H, W)
        
        # 3. Early Fusion (Channel Concatenation)
        # 원본 RGB(정규화됨)와 SRM 노이즈를 합침
        combined = torch.cat([x, srm_x], dim=1) # (Batch, 6, H, W)
        
        # 4. 백본 통과
        logits = self.model(combined)
        return logits


def load_model(checkpoint_path: str, model_name: str = "tf_efficientnetv2_m.in21k", device: str = "cuda") -> DeepfakeDetector:
    """
    체크포인트에서 DeepfakeDetector 모델 로드
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = DeepfakeDetector(model_name=model_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model
