from ultralytics import YOLO
import torch


def train_yolo_models():
    """
    Script to sequentially train YOLOv11 models.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    models_to_train = [
        'yolo11n.pt',
        'yolo11s.pt',
        'yolo11m.pt',
        'yolo11l.pt',
        'yolo11x.pt',
    ]

    training_params = {
        'data': 'radiolaria_dataset.yaml',
        'epochs': 100,
        'batch': 32,
        'imgsz': 640,
        'device': device,
        'workers': 8
    }

    for model_name in models_to_train:
        print(f"\n--- Training model: {model_name} ---")
        try:
            model = YOLO(model_name)
            run_name = f"{model_name.split('.')[0]}_radiolaria_train"

            model.train(
                **training_params,
                name=run_name
            )

            print(f"Successfully trained {model_name}.")
            print(f"Results saved to 'runs/detect/{run_name}'")

        except Exception as e:
            print(f"Error training model {model_name}: {e}")
            continue

    print("\nAll training tasks completed!")


if __name__ == '__main__':
    train_yolo_models()