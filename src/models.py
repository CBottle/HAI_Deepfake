"""
모델 정의 모듈

이 모듈에서는 딥페이크 탐지를 위한 모델 아키텍처를 정의합니다.
"""

import torch
import torch.nn as nn
from transformers import ViTForImageClassification


class DeepfakeDetector(nn.Module):
    """
    딥페이크 탐지 모델

    Args:
        model_name (str): 사전학습 모델 이름
        num_classes (int): 출력 클래스 수 (기본: 2 - Real/Fake)
        pretrained (bool): 사전학습 가중치 사용 여부
    """

    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        num_classes: int = 2,
        pretrained: bool = True
    ):
        super().__init__()

        # ViT 모델 로드
        if pretrained:
            self.model = ViTForImageClassification.from_pretrained(
                model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
        else:
            from transformers import ViTConfig
            config = ViTConfig.from_pretrained(model_name)
            config.num_labels = num_classes
            self.model = ViTForImageClassification(config)

    def forward(self, pixel_values):
        """
        순전파

        Args:
            pixel_values: 입력 이미지 텐서

        Returns:
            logits: 모델 출력 (batch_size, num_classes)
        """
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits


def load_model(checkpoint_path: str, device: str = "cuda") -> DeepfakeDetector:
    """
    체크포인트에서 모델 로드

    Args:
        checkpoint_path: 체크포인트 파일 경로
        device: 디바이스 (cuda/cpu)

    Returns:
        로드된 모델
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 모델 초기화
    model = DeepfakeDetector()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model
