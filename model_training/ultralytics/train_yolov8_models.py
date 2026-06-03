# -*- coding: utf-8 -*-

"""
Script for batch training YOLOv8 models (n, s, m, l, x).
"""

from ultralytics import YOLO
import os


def main():
    data_config_path = "radiolaria.yaml"

    if not os.path.exists(data_config_path):
        print(f"Error: Dataset config '{data_config_path}' not found.")
        return

    models_to_train = [
        'yolov8n.pt',
        'yolov8s.pt',
        'yolov8m.pt',
        'yolov8l.pt',
        'yolov8x.pt'
    ]

    train_params = {
        'epochs': 100,
        'batch': 32,
        'workers': 8,
        'project': 'runs/train_radiolaria',
    }

    print(f"Starting batch training with params: {train_params}")

    for model_name in models_to_train:
        model_short_name = model_name.split('.')[0]
        print(f"\n--- Training {model_name} ---")

        try:
            model = YOLO(model_name)

            model.train(
                data=data_config_path,
                epochs=train_params['epochs'],
                batch=train_params['batch'],
                workers=train_params['workers'],
                project=train_params['project'],
                name=model_short_name,
                exist_ok=False
            )

            print(f"Successfully trained {model_name}.")
            print(f"Results saved to: {train_params['project']}/{model_short_name}")

        except Exception as e:
            print(f"Error training model {model_name}: {e}")

    print("\nAll batch training tasks completed.")


if __name__ == '__main__':
    main()