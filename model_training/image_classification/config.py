# -*- coding: utf-8 -*-
# Configuration parameters

import os

# Base directory (where this config file is located)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)

# Path configuration (relative to model_training/ directory)
RAW_DATA_DIR = os.path.join(PARENT_DIR, "classification_dataset")
TRAIN_DATA_DIR = os.path.join(RAW_DATA_DIR, "train")
VAL_DATA_DIR = os.path.join(RAW_DATA_DIR, "val")
PRETRAINED_WEIGHTS_DIR = os.path.join(BASE_DIR, "pretrained_weights")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
LOGS_DIR = os.path.join(BASE_DIR, "training_logs")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")

# Training hyperparameters
NUM_EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_WORKERS = 4

# Train/val split ratio
TRAIN_VAL_SPLIT = 0.8

# Model list
MODELS_TO_TRAIN = [
    'squeezenet1_1', 'shufflenet_v2_x1_0', 'mobilenet_v2', 'mobilenet_v3_large',
    'resnet18', 'efficientnet_b0', 'densenet121', 'efficientnet_b2',
    'resnet50', 'resnext50_32x4d', 'wide_resnet50_2', 'vgg16',
]
