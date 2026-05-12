# -*- coding: utf-8 -*-
# Standalone model evaluation script

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    accuracy_score, confusion_matrix, classification_report
)

from config import VAL_DATA_DIR, SAVED_MODELS_DIR, LOGS_DIR, BATCH_SIZE, NUM_WORKERS
from models import get_model
from dataset import get_transforms
from torch.utils.data import DataLoader
from torchvision import datasets


def evaluate_model(model_path, model_name, num_classes):
    """Load model and evaluate on validation set, return multi-metric results"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nEvaluating {model_name} on {device}")
    
    # Load model
    model = get_model(model_name, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Load validation data
    val_transform = get_transforms()['val']
    val_dataset = datasets.ImageFolder(VAL_DATA_DIR, val_transform)
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS
    )
    
    class_names = val_dataset.classes
    
    # Inference
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    cm = confusion_matrix(all_labels, all_preds)
    
    print(f"\nResults for {model_name}:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(cm)
    print(f"\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
    
    return {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm.tolist()
    }


def main():
    print("=" * 50)
    print("Model Evaluation on Validation Set")
    print("=" * 50)
    
    # Get class information
    val_dataset = datasets.ImageFolder(VAL_DATA_DIR, get_transforms()['val'])
    class_names = val_dataset.classes
    num_classes = len(class_names)
    print(f"Classes: {class_names}")
    
    # Find all trained models
    model_files = [f for f in os.listdir(SAVED_MODELS_DIR) if f.endswith('.pth')]
    
    if not model_files:
        print(f"No trained models found in {SAVED_MODELS_DIR}")
        return
    
    results = []
    for mf in model_files:
        model_name = mf.replace('best_classifier_', '').replace('.pth', '')
        model_path = os.path.join(SAVED_MODELS_DIR, mf)
        
        try:
            result = evaluate_model(model_path, model_name, num_classes)
            results.append(result)
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
    
    # Save evaluation results
    if results:
        os.makedirs(LOGS_DIR, exist_ok=True)
        results_df = pd.DataFrame(results)
        results_df = results_df[['model', 'accuracy', 'precision', 'recall', 'f1']]
        results_df.to_csv(os.path.join(LOGS_DIR, 'evaluation_results.csv'), index=False)
        
        print("\n" + "=" * 50)
        print("Evaluation Summary")
        print("=" * 50)
        print(results_df.to_string(index=False))
        print(f"\nResults saved to {LOGS_DIR}/evaluation_results.csv")


if __name__ == '__main__':
    main()
