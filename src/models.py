"""
모델 정의 모듈

이 모듈에서는 딥페이크 탐지를 위한 모델 아키텍처를 정의합니다.
"""

import torch
import torch.nn as nn
import timm  # PyTorch Image Models (EfficientNet 등을 위해 사용)


class DeepfakeDetector(nn.Module):
    """
    딥페이크 탐지 모델 (EfficientNet-V2 기반)

    Args:
        model_name (str): 사전학습 모델 이름 (기본: tf_efficientnetv2_m.in21k)
        num_classes (int): 출력 클래스 수 (기본: 2 - Real/Fake)
        pretrained (bool): 사전학습 가중치 사용 여부
    """

    def __init__(
        self,
        model_name: str = "tf_efficientnetv2_m.in21k",
        num_classes: int = 2,
        pretrained: bool = True
    ):
        super().__init__()

        # timm을 사용하여 모델 로드 (EfficientNet, ResNet, ViT 등 모두 지원)
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )

    def forward(self, pixel_values):
        """
        순전파

        Args:
            pixel_values: 입력 이미지 텐서

        Returns:
            logits: 모델 출력 (batch_size, num_classes) - Tensor 반환
        """
        # timm 모델은 logits 텐서를 바로 반환합니다.
        logits = self.model(pixel_values)
        return logits


def load_model(checkpoint_path: str, model_name: str = "tf_efficientnetv2_m.in21k", device: str = "cuda") -> DeepfakeDetector:
    """
    체크포인트에서 모델 로드

    Args:
        checkpoint_path: 체크포인트 파일 경로
        model_name: 모델 이름 (config와 일치해야 함)
        device: 디바이스 (cuda/cpu)

    Returns:
        로드된 모델
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 모델 초기화 (전달받은 model_name 사용)
    model = DeepfakeDetector(model_name=model_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model
