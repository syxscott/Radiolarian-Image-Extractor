# -*- coding: utf-8 -*-
# Model factory functions

import os
import torch
import torch.nn as nn
from torchvision import models
from config import PRETRAINED_WEIGHTS_DIR

def get_model(model_name, num_classes):
    """Create model instance with pretrained weights and adjusted classifier head"""
    
    print(f"\n{'=' * 20}")
    print(f"Configuring model: {model_name}")
    print(f"{'=' * 20}")
    
    os.makedirs(PRETRAINED_WEIGHTS_DIR, exist_ok=True)
    local_weights_path = os.path.join(PRETRAINED_WEIGHTS_DIR, f"{model_name}_pretrained.pth")
    
    weights_map = {
        'resnet18': models.ResNet18_Weights.IMAGENET1K_V1,
        'resnet50': models.ResNet50_Weights.IMAGENET1K_V2,
        'vgg16': models.VGG16_Weights.IMAGENET1K_V1,
        'densenet121': models.DenseNet121_Weights.IMAGENET1K_V1,
        'mobilenet_v2': models.MobileNet_V2_Weights.IMAGENET1K_V2,
        'mobilenet_v3_large': models.MobileNet_V3_Large_Weights.IMAGENET1K_V2,
        'efficientnet_b0': models.EfficientNet_B0_Weights.IMAGENET1K_V1,
        'efficientnet_b2': models.EfficientNet_B2_Weights.IMAGENET1K_V1,
        'resnext50_32x4d': models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2,
        'wide_resnet50_2': models.Wide_ResNet50_2_Weights.IMAGENET1K_V2,
        'shufflenet_v2_x1_0': models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1,
        'squeezenet1_1': models.SqueezeNet1_1_Weights.IMAGENET1K_V1,
    }
    
    model_constructor = getattr(models, model_name)
    
    # Load weights
    if os.path.exists(local_weights_path):
        print(f"Loading local weights: {local_weights_path}")
        model = model_constructor(weights=None)
        model.load_state_dict(torch.load(local_weights_path, map_location='cpu'))
    else:
        print(f"Downloading pretrained weights...")
        weights = weights_map.get(model_name)
        if weights:
            model = model_constructor(weights=weights)
            torch.save(model.state_dict(), local_weights_path)
            print(f"Weights saved to: {local_weights_path}")
        else:
            model = model_constructor(weights=None)
            print(f"No pretrained weights, training from scratch")
    
    # Adjust classifier head
    if 'resnet' in model_name or 'resnext' in model_name or 'wide_resnet' in model_name:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif 'vgg' in model_name or 'densenet' in model_name:
        last_layer = list(model.classifier.children())[-1]
        num_ftrs = last_layer.in_features
        model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
    elif 'mobilenet' in model_name:
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
    elif 'efficientnet' in model_name:
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    elif 'squeezenet' in model_name:
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model.num_classes = num_classes
    elif 'shufflenet' in model_name:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    print(f"Classifier head modified to output {num_classes} classes")
    return model
