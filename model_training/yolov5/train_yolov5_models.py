# -*- coding: utf-8 -*-

"""
Script for batch training YOLOv5 models (n, s, m, l, x).
Requires the YOLOv5 repository structure and pre-trained weights in the working directory.
"""

import os
import subprocess
import sys


def main():
    # Set YOLOv5 directory path
    YOLOV5_DIR = '.'

    yolov5_train_script = os.path.join(YOLOV5_DIR, 'train.py')
    data_config_path = "radiolaria.yaml"

    # Validate paths
    if not os.path.exists(yolov5_train_script):
        print(f"Error: YOLOv5 train script not found at '{yolov5_train_script}'.")
        return

    if not os.path.exists(data_config_path):
        print(f"Error: Dataset config '{data_config_path}' not found.")
        return

    models_to_train = [
        'yolov5n.pt',
        'yolov5s.pt',
        'yolov5m.pt',
        'yolov5l.pt',
        'yolov5x.pt'
    ]

    # Validate pre-trained weights
    for model_pt in models_to_train:
        if not os.path.exists(model_pt):
            print(f"Error: Pre-trained weights '{model_pt}' not found in the current directory.")
            return

    train_params = {
        'imgsz': 640,
        'epochs': 100,
        'batch_size': 32,
        'workers': 8,
        'project': 'runs/train_radiolaria_v5',
    }

    print(f"Starting batch training with params: {train_params}")

    for model_weights in models_to_train:
        model_short_name = model_weights.split('.')[0]
        print(f"\n--- Training {model_weights} ---")

        command = [
            sys.executable,
            yolov5_train_script,
            '--weights', model_weights,
            '--data', data_config_path,
            '--epochs', str(train_params['epochs']),
            '--batch-size', str(train_params['batch_size']),
            '--workers', str(train_params['workers']),
            '--imgsz', str(train_params['imgsz']),
            '--project', train_params['project'],
            '--name', model_short_name,
            '--exist-ok'
        ]

        try:
            subprocess.run(command, check=True)
            print(
                f"Successfully trained {model_weights}. Results saved to: {train_params['project']}/{model_short_name}")
        except subprocess.CalledProcessError as e:
            print(f"Subprocess error while training {model_weights}: {e}")
        except Exception as e:
            print(f"Unknown error while training {model_weights}: {e}")

    print("\nAll batch training tasks completed.")


if __name__ == '__main__':
    main()