# -*- coding: utf-8 -*-
"""
@file: config.py
@description: Global configuration settings, constants, and path definitions.
              Centralizes all hardcoded values to ensure easy maintenance.
@author: Yaxuan Shen
@date: 2025-10-01
"""

import os
from collections import OrderedDict
import torch


class Config:
    """
    Global configuration state for the Radiolarian Analysis Tool.

    ----------------------------------------------------------------
    MODEL FILE NAMING CONVENTION (for `models/classifiers/`)
    ----------------------------------------------------------------
    The application auto-detects the architecture of a `.pth` weight
    file by fuzzy-matching its FILENAME against the architecture
    keywords listed in `CLASSIFIER_ARCH_KEYWORDS` below.  The first
    keyword that appears (case-insensitive) in the filename determines
    the architecture that will be constructed when loading the weights.

    Therefore, weight files should be named so that they include one
    of the supported keywords.  Examples of good / bad names:

        GOOD  :  resnet50_radiolarian_run1.pth
        GOOD  :  best_mobilenet_v2_finetuned.pth
        GOOD  :  efficientnet_b0_v3.pth
        BAD   :  classifier_run3.pth            (no architecture keyword)
        BAD   :  model.pth                       (no architecture keyword)

    If a file's architecture cannot be determined, it is silently
    skipped and a warning is printed to the console.
    ----------------------------------------------------------------
    """

    # ---------------------------------------------------------
    # System Paths
    # ---------------------------------------------------------
    # Resolve relative to THIS file so the app behaves the same no
    # matter which working directory it is launched from.
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
    # YOLO / Classifier models are discovered dynamically at runtime via
    # `scan_yolo_models()` and `scan_classifier_models()` from the model
    # directories — no hard-coded lists.

    # Classification Classes
    CLASSIFIER_CLASS_NAMES = ['non_radiolarian', 'radiolarian']
    POSITIVE_CLASS_NAME = 'radiolarian'

    # Supported classifier architectures, ordered by fuzzy-match priority
    # (longer / more specific keywords MUST come first so that
    # "resnet18" is not shadowed by "resnet", "mobilenet_v3_small" not
    # shadowed by "mobilenet_v3", etc.)
    CLASSIFIER_ARCH_KEYWORDS = [
        ('efficientnet_b2',     'efficientnet_b2'),
        ('efficientnet_b0',     'efficientnet_b0'),
        ('mobilenet_v3_small',  'mobilenet_v3_small'),
        ('mobilenet_v3_large',  'mobilenet_v3_large'),
        ('mobilenet_v3',        'mobilenet_v3_large'),  # fallback
        ('mobilenet_v2',        'mobilenet_v2'),
        ('wide_resnet50',       'wide_resnet50_2'),
        ('resnext50',           'resnext50_32x4d'),
        ('resnet18',            'resnet18'),
        ('resnet50',            'resnet50'),
        ('densenet121',         'densenet121'),
        ('shufflenet',          'shufflenet_v2_x1_0'),
        ('vgg16',               'vgg16'),
        ('squeezenet',          'squeezenet1_1'),
    ]

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

    # Common DPI presets shown in the UI.  These are *real* DPI values;
    # the PDF worker internally converts them to a fitz zoom factor
    # of `dpi / 72`.
    DPI_PRESETS = (72, 150, 200, 300, 400, 600, 800, 1200)
    DEFAULT_DPI = 300

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

    # ---------------------------------------------------------
    # Dynamic Model Discovery
    # ---------------------------------------------------------
    @staticmethod
    def scan_yolo_models():
        """
        Dynamically scan `models/yolo/` for `.pt` weights.

        Skips dotfiles (Linux/macOS hidden-file convention) so that
        editor backups like `.foo.pt.swp` or `._yolo11x.pt` (macOS
        AppleDouble) don't appear in the dropdown.

        Returns:
            OrderedDict: {display_name (filename without ext) -> absolute_path}
            Display name is the bare filename (e.g. "yolo11x") so the user
            can immediately see what file lives in the dropdown.
        """
        folder = os.path.join(Config.MODELS_DIR, "yolo")
        os.makedirs(folder, exist_ok=True)
        result = OrderedDict()
        try:
            entries = sorted(os.listdir(folder))
        except OSError:
            entries = []
        for fname in entries:
            # Skip Linux/macOS hidden / metadata files
            if fname.startswith('.'):
                continue
            if not fname.lower().endswith('.pt'):
                continue
            full = os.path.join(folder, fname)
            if not os.path.isfile(full):
                continue
            result[os.path.splitext(fname)[0]] = full
        return result

    @staticmethod
    def _match_classifier_arch(filename: str):
        """
        Fuzzy-match a classifier filename to a supported architecture key.
        Returns the architecture key (e.g. 'resnet50') or None if unknown.

        Matching is keyword-based, in priority order defined by
        `CLASSIFIER_ARCH_KEYWORDS` (most specific keyword first).
        Example:
            "best_radiolarian_classifier_resnet50_torchvision.pth" -> 'resnet50'
            "mobilenet_v2_custom.pth"                              -> 'mobilenet_v2'
            "effb0_run3.pth"                                       -> 'efficientnet_b0'
        """
        fn = filename.lower()
        for keyword, arch_key in Config.CLASSIFIER_ARCH_KEYWORDS:
            if keyword in fn:
                return arch_key
        return None

    @staticmethod
    def scan_classifier_models():
        """
        Dynamically scan `models/classifiers/` for `.pth` weights and
        determine each file's architecture by fuzzy filename matching.

        Skips dotfiles (Linux/macOS hidden-file convention) so that
        editor backups and OS-metadata files don't appear in the dropdown.

        Returns:
            OrderedDict: {display_name -> (file_path, arch_key)}
            Files whose architecture cannot be determined are SKIPPED and
            logged to stderr so that the user knows a file was ignored.
        """
        folder = os.path.join(Config.MODELS_DIR, "classifiers")
        os.makedirs(folder, exist_ok=True)
        result = OrderedDict()
        try:
            entries = sorted(os.listdir(folder))
        except OSError:
            entries = []
        for fname in entries:
            # Skip Linux/macOS hidden / metadata files
            if fname.startswith('.'):
                continue
            if not fname.lower().endswith('.pth'):
                continue
            full = os.path.join(folder, fname)
            if not os.path.isfile(full):
                continue
            arch_key = Config._match_classifier_arch(fname)
            if arch_key is None:
                print(f"[Config] Skipping unknown classifier: {fname} "
                      f"(no matching architecture keyword)")
                continue
            result[os.path.splitext(fname)[0]] = (full, arch_key)
        return result