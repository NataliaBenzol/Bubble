from __future__ import annotations

import torch.nn as nn

from .backbones import create_backbone


def create_resnet50_model(num_classes: int = 7, pretrained: bool = True) -> nn.Module:
    return create_backbone("resnet50", num_classes=num_classes)