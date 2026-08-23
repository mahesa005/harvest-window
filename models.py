#!/usr/bin/env python3
"""
HarvestWindow — Model 2 backbone factory

Shared by train_model2.py and evaluate_model2.py so neither imports
from the other.

mobilenet_v3_large added to support externally-trained weights (a
teammate trained outside this codebase, different checkpoint format —
see checkpoint_adapter.py). Its classifier head structure differs from
mobilenet_v2/efficientnet_b0: final layer is classifier[3], not
classifier[1] — torchvision's MobileNetV3 classifier is
Sequential(Linear, Hardswish, Dropout, Linear), not the simpler
Sequential(Dropout, Linear) the other two use.
"""

import torch.nn as nn
from torchvision import models


def build_model(backbone: str, num_classes: int, pretrained: bool = True):
    """
    pretrained=True (default): loads ImageNet weights first, for actual
    training runs where that head-start matters.
    pretrained=False: skips the download entirely — use this when you're
    about to load a full trained state_dict anyway (e.g. checkpoint_adapter.py),
    since downloading ImageNet weights just to immediately overwrite them
    is pure waste.
    """
    if backbone == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif backbone == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif backbone == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(
            f"Unknown backbone '{backbone}' — use 'mobilenet_v2', 'efficientnet_b0', or 'mobilenet_v3_large'"
        )
    return model
