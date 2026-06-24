from __future__ import annotations

from typing import Callable

import torch.nn as nn
from torchvision import models


def _make_head(num_features: int, num_classes: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )


def _resnet50(num_classes: int) -> nn.Module:
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = _make_head(model.fc.in_features, num_classes)
    return model


def _vgg16(num_classes: int) -> nn.Module:
    weights = models.VGG16_Weights.IMAGENET1K_V1
    model = models.vgg16(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)
    return model


_REGISTRY: dict[str, Callable[[int], nn.Module]] = {
    "resnet50": _resnet50,
    "vgg16": _vgg16,
}


def create_backbone(name: str, num_classes: int) -> nn.Module:
    key = name.lower()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown backbone: {name!r}. Available: {available}")
    return _REGISTRY[key](num_classes)


def available_backbones() -> list[str]:
    return sorted(_REGISTRY)


def register_backbone(name: str, factory: Callable[[int], nn.Module]) -> None:
    _REGISTRY[name.lower()] = factory