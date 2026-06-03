# -*- coding: utf-8 -*-
# Data loading and augmentation

import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config import TRAIN_DATA_DIR, VAL_DATA_DIR, BATCH_SIZE, NUM_WORKERS

def get_transforms():
    """Return train and val data augmentation pipelines"""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return {'train': train_transform, 'val': val_transform}

def create_dataloaders():
    """Create train and validation dataloaders"""
    transforms_dict = get_transforms()
    
    print(f"Train data dir: {TRAIN_DATA_DIR}")
    print(f"Val data dir: {VAL_DATA_DIR}")
    
    # Debug: print directory structure
    print("\n[TRAIN] Directory structure:")
    if os.path.exists(TRAIN_DATA_DIR):
        for item in os.listdir(TRAIN_DATA_DIR):
            item_path = os.path.join(TRAIN_DATA_DIR, item)
            if os.path.isdir(item_path):
                files = os.listdir(item_path)
                print(f"  {item}/ ({len(files)} files)")
    else:
        print(f"  ERROR: {TRAIN_DATA_DIR} does not exist!")
    
    print("\n[VAL] Directory structure:")
    if os.path.exists(VAL_DATA_DIR):
        for item in os.listdir(VAL_DATA_DIR):
            item_path = os.path.join(VAL_DATA_DIR, item)
            if os.path.isdir(item_path):
                files = os.listdir(item_path)
                print(f"  {item}/ ({len(files)} files)")
    else:
        print(f"  ERROR: {VAL_DATA_DIR} does not exist!")
    
    train_dataset = datasets.ImageFolder(TRAIN_DATA_DIR, transforms_dict['train'])
    val_dataset = datasets.ImageFolder(VAL_DATA_DIR, transforms_dict['val'])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    dataset_sizes = {
        'train': len(train_dataset),
        'val': len(val_dataset)
    }
    
    class_names = train_dataset.classes
    
    return train_loader, val_loader, dataset_sizes, class_names

def compute_class_weights(loader, num_classes, device):
    """Compute class weights for CrossEntropyLoss based on training set distribution.

    Previously this iterated the entire DataLoader just to read labels,
    which (a) loaded and augmented every image, wasting an epoch's worth
    of IO+CPU, and (b) advanced the random state used by RandomResizedCrop /
    RandomHorizontalFlip / etc., subtly changing the actual training
    distribution.

    The fix: ImageFolder exposes `targets` (a flat list of integer labels)
    directly — counting them takes microseconds and touches no images.
    Fallback to the slow path only for non-ImageFolder datasets.
    """
    dataset = loader.dataset
    class_counts = [0] * num_classes

    if hasattr(dataset, 'targets'):
        # Fast path — ImageFolder / DatasetFolder
        for t in dataset.targets:
            class_counts[int(t)] += 1
    else:
        # Slow fallback — generic Dataset without targets attribute
        for _, labels in loader:
            for label in labels:
                class_counts[label.item()] += 1

    # Inverse weight: n_samples / (n_classes * n_samples_of_class)
    # Guard against empty classes to avoid ZeroDivisionError.
    total = sum(class_counts)
    weights = [
        (total / (num_classes * count)) if count > 0 else 0.0
        for count in class_counts
    ]

    print(f"Class distribution: {class_counts}")
    print(f"Class weights: {[f'{w:.2f}' for w in weights]}")

    return torch.tensor(weights, dtype=torch.float32).to(device)
