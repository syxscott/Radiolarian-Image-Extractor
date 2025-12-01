# -*- coding: utf-8 -*-
"""
@file: config.py
@description: Global configuration settings, constants, and path definitions.
              Centralizes all hardcoded values to ensure easy maintenance.
@author: Yaxuan Shen
@date: 2025-10-01
"""

import os
import torch


class Config:
    """
    Global configuration state for the Radiolarian Analysis Tool.
    """

    # ---------------------------------------------------------
    # System Paths
    # ---------------------------------------------------------
    BASE_DIR = os.getcwd()

    # Input/Output Directories
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "01_Source_PDFs")
    PAGES_FOLDER = os.path.join(BASE_DIR, "02_Processed_Pages")
    CROPPED_FOLDER = os.path.join(BASE_DIR, "03_Candidate_Images")
    REJECTED_FOLDER = os.path.join(BASE_DIR, "03a_Rejected_Images")
    LOGS_FOLDER = os.path.join(BASE_DIR, "05_Logs")
    MODELS_DIR = os.path.join(BASE_DIR, "models")

    # ---------------------------------------------------------
    # Model Configurations
    # ---------------------------------------------------------

    # Dictionary mapping display names to file paths for YOLO models
    YOLO_MODELS = {
        "yolo11x.pt": os.path.join(MODELS_DIR, "yolo", "yolo11x.pt"),
        "yolov8n.pt": os.path.join(MODELS_DIR, "yolo", "yolov8n.pt"),
        "yolov8s-custom.pt": os.path.join(MODELS_DIR, "yolo", "yolov8s-custom.pt")
    }

    # Dictionary for Classifier models (ResNet/MobileNet)
    CLASSIFIER_MODELS = {
        "resnet50_torchvision.pth": os.path.join(MODELS_DIR, "classifiers",
                                                 "best_radiolarian_classifier_resnet50_torchvision.pth"),
        "mobilenet_v2.pth": os.path.join(MODELS_DIR, "classifiers", "mobilenet_v2_custom.pth")
    }

    # Classification Classes
    CLASSIFIER_CLASS_NAMES = ['non_radiolarian', 'radiolarian']
    POSITIVE_CLASS_NAME = 'radiolarian'

    # ---------------------------------------------------------
    # Processing Parameters
    # ---------------------------------------------------------
    ALLOWED_EXTENSIONS = {'.pdf'}

    # Hardware Acceleration Detection
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_GPUS = torch.cuda.device_count()
    CPU_COUNT = os.cpu_count() or 4

    # Threading Limits (Auto-tuned based on hardware)
    # PDF conversion is memory intensive, limit threads carefully
    PDF_MAX_WORKERS = min(int(CPU_COUNT * 1.5), 16)

    # GPU workers can be higher if VRAM allows
    GPU_MAX_WORKERS = NUM_GPUS * 4 if NUM_GPUS > 0 else 2

    # Batch processing sizes
    DETECTION_BATCH_SIZE = 16
    CLASSIFICATION_BATCH_SIZE = 32

    @staticmethod
    def ensure_directories():
        """Creates necessary directory structure if it doesn't exist."""
        dirs = [
            Config.UPLOAD_FOLDER, Config.PAGES_FOLDER, Config.CROPPED_FOLDER,
            Config.REJECTED_FOLDER, Config.LOGS_FOLDER,
            os.path.join(Config.MODELS_DIR, "yolo"),
            os.path.join(Config.MODELS_DIR, "classifiers")
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)