#!/usr/bin/env python3
"""
HarvestWindow — Model 2 backbone factory

Shared by train_model2.py and evaluate_model2.py so neither imports
from the other.
"""

import torch.nn as nn
from torchvision import models


def build_model(backbone: str, num_classes: int):
    if backbone == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unknown backbone '{backbone}' — use 'mobilenet_v2' or 'efficientnet_b0'")
    return model
