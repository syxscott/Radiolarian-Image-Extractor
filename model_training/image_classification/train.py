# -*- coding: utf-8 -*-
# Training script: supports class weights and multi-metric evaluation

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

from config import NUM_EPOCHS, LEARNING_RATE, SAVED_MODELS_DIR, LOGS_DIR, CHECKPOINTS_DIR
from models import get_model
from dataset import create_dataloaders, compute_class_weights


def train_one_model(model, model_name, train_loader, val_loader, dataset_sizes, num_classes):
    """Train single model with class weights and multi-metric evaluation"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Multi-GPU support
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    
    # Compute class weights for imbalanced data (Reviewer 2 requirement)
    class_weights = compute_class_weights(train_loader, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    # Resume training from checkpoint
    start_epoch = 0
    best_f1 = 0.0
    training_history = []
    checkpoint_path = os.path.join(CHECKPOINTS_DIR, f"checkpoint_{model_name}.pth")
    log_path = os.path.join(LOGS_DIR, f"{model_name}_training_log.csv")
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        state_dict = checkpoint['model_state_dict']
        if isinstance(model, nn.DataParallel) and not list(state_dict.keys())[0].startswith('module.'):
            state_dict = {'module.' + k: v for k, v in state_dict.items()}
        elif not isinstance(model, nn.DataParallel) and list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_f1 = checkpoint.get('best_f1', 0.0)
        
        if os.path.exists(log_path):
            training_history = pd.read_csv(log_path).to_dict('records')
        
        print(f"Resuming from epoch {start_epoch}, best F1: {best_f1:.4f}")
    
    start_time = time.time()
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\n[{model_name}] Epoch {epoch + 1}/{NUM_EPOCHS}")
        epoch_log = {'epoch': epoch + 1}
        
        # Training phase
        model.train()
        train_loss, train_corrects = 0.0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            train_corrects += torch.sum(preds == labels.data)
        
        scheduler.step()
        
        train_loss = train_loss / dataset_sizes['train']
        train_acc = train_corrects.double() / dataset_sizes['train']
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        
        epoch_log['train_loss'] = train_loss
        epoch_log['train_acc'] = train_acc.item()
        
        # Validation phase: compute multi-metric evaluation (Reviewer 2 requirement)
        model.eval()
        val_loss, val_corrects = 0.0, 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss = val_loss / dataset_sizes['val']
        val_acc = val_corrects.double() / dataset_sizes['val']
        
        # Compute Precision, Recall, F1 (macro average, zero_division=0)
        val_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        val_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, "
              f"P: {val_precision:.4f}, R: {val_recall:.4f}, F1: {val_f1:.4f}")
        
        epoch_log['val_loss'] = val_loss
        epoch_log['val_acc'] = val_acc.item()
        epoch_log['val_precision'] = val_precision
        epoch_log['val_recall'] = val_recall
        epoch_log['val_f1'] = val_f1
        
        # Save best model (based on F1 instead of Accuracy only)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_path = os.path.join(SAVED_MODELS_DIR, f"best_classifier_{model_name}.pth")
            os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
            save_model = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(save_model.state_dict(), best_model_path)
            print(f"New best model saved! F1: {best_f1:.4f}")
        
        training_history.append(epoch_log)
        
        # Save checkpoint
        os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
        save_model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save({
            'epoch': epoch,
            'model_state_dict': save_model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_f1': best_f1,
        }, checkpoint_path)
        
        # Save log
        os.makedirs(LOGS_DIR, exist_ok=True)
        log_df = pd.DataFrame(training_history)
        log_df.to_csv(log_path, index=False)
    
    elapsed = time.time() - start_time
    print(f"\n{model_name} training completed in {elapsed // 60:.0f}m {elapsed % 60:.0f}s")
    print(f"Best Val F1: {best_f1:.4f}")
    
    return best_f1, elapsed, log_path


def main():
    print("=" * 50)
    print("Radiolarian Image Classification Training")
    print("=" * 50)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    
    train_loader, val_loader, dataset_sizes, class_names = create_dataloaders()
    num_classes = len(class_names)
    
    print(f"\nClasses: {class_names}")
    print(f"Train size: {dataset_sizes['train']}, Val size: {dataset_sizes['val']}")
    
    from config import MODELS_TO_TRAIN
    results = {}
    
    for model_name in MODELS_TO_TRAIN:
        print(f"\n{'#' * 50}")
        print(f"Training: {model_name}")
        print(f"{'#' * 50}")
        
        try:
            model = get_model(model_name, num_classes)
            best_f1, training_time, log_file = train_one_model(
                model, model_name, train_loader, val_loader, dataset_sizes, num_classes
            )
            
            results[model_name] = {
                'Best_F1': f"{best_f1:.4f}",
                'Time_min': f"{training_time / 60:.2f}",
                'Log': log_file
            }
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            results[model_name] = {'Best_F1': 'Error', 'Time_min': 'Error', 'Log': 'N/A'}
    
    print("\n" + "=" * 50)
    print("Training Results Summary")
    print("=" * 50)
    results_df = pd.DataFrame.from_dict(results, orient='index').reindex(MODELS_TO_TRAIN)
    print(results_df)
    
    report_path = "model_comparison_report.csv"
    results_df.to_csv(report_path)
    print(f"\nReport saved to: {report_path}")


if __name__ == '__main__':
    main()
