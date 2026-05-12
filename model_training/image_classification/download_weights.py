# -*- coding: utf-8 -*-
# Download pretrained model weights

import os
import torch
from torchvision import models

# Weights will be saved to this directory
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained_weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# Note: Using weights= parameter, torch downloads to cache automatically
# Then we copy from cache to local directory

MODELS_CONFIG = {
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

def get_torch_cache_dir():
    """Get torch hub cache directory"""
    return torch.hub.get_dir()

def download_weights(model_name=None):
    """Download pretrained weights for specified model or all models"""
    
    print(f"Weights will be saved to: {WEIGHTS_DIR}")
    print(f"Torch cache dir: {get_torch_cache_dir()}")
    print("=" * 50)
    
    if model_name:
        if model_name not in MODELS_CONFIG:
            print(f"Error: Unknown model '{model_name}'")
            return
        models_to_download = [model_name]
    else:
        models_to_download = list(MODELS_CONFIG.keys())
    
    for model_name in models_to_download:
        weights_file = os.path.join(WEIGHTS_DIR, f"{model_name}_pretrained.pth")
        
        if os.path.exists(weights_file):
            print(f"[SKIP] {model_name}: already exists")
            continue
        
        print(f"[DOWNLOAD] {model_name}...")
        try:
            # Load model with weights (torch will download to cache)
            model_constructor = getattr(models, model_name)
            model = model_constructor(weights=MODELS_CONFIG[model_name])
            
            # Save to local directory
            torch.save(model.state_dict(), weights_file)
            size_mb = os.path.getsize(weights_file) / (1024 * 1024)
            print(f"[DONE] {model_name}: saved to {weights_file} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"[ERROR] {model_name}: {e}")
    
    print("=" * 50)
    print("Download complete!")

def list_weights():
    """List downloaded weights"""
    print("Downloaded weights:")
    for f in os.listdir(WEIGHTS_DIR):
        if f.endswith('.pth'):
            size_mb = os.path.getsize(os.path.join(WEIGHTS_DIR, f)) / (1024 * 1024)
            print(f"  {f} ({size_mb:.1f} MB)")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Download pretrained model weights')
    parser.add_argument('--model', type=str, default=None, help='Model name to download (default: all)')
    args = parser.parse_args()
    
    download_weights(args.model)
    print("\nCurrent weights in directory:")
    list_weights()
