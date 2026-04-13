import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, f1_score, roc_auc_score
import copy
import os
import importlib

# Standard trick to import scripts starting with numbers
seq_module = importlib.import_module("04_sequences")
model_module = importlib.import_module("05_model")

train_loader = seq_module.train_loader
val_loader = seq_module.val_loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize Model
model = model_module.MultiTaskModel(n_features=18, window_size=30).to(device)

# Tasks: Optimizer, Scheduler, Early Stopping
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4

optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Early Stopping configs
ES_PATIENCE = 10
best_val_loss = float('inf')
best_model_weights = copy.deepcopy(model.state_dict())
es_counter = 0

# Loss weights
# MSE is usually large (e.g., 20^2 = 400), BCE is usually small (~0.6). 
# We boost beta to make the alert loss comparable during tuning.
ALPHA = 1.0  
BETA = 100.0 

history = {
    'train_loss': [], 'val_loss': [], 'val_rmse': [], 'val_f1': [], 'val_auc': []
}

print("\n--- Starting Training Loop ---")

for epoch in range(EPOCHS):
    # --- TRAINING ---
    model.train()
    train_loss_epoch = 0.0
    
    for X_batch, rul_batch, alert_batch in train_loader:
        X_batch = X_batch.to(device)
        rul_batch = rul_batch.to(device)
        alert_batch = alert_batch.to(device)
        
        optimizer.zero_grad()
        rul_pred, alert_pred = model(X_batch)
        
        total_loss, mse, bce = model_module.combined_loss(
            rul_pred, rul_batch, alert_pred, alert_batch, alpha=ALPHA, beta=BETA)
        
        total_loss.backward()
        optimizer.step()
        
        train_loss_epoch += total_loss.item() * X_batch.size(0)
        
    train_loss_epoch /= len(train_loader.dataset)
    
    # --- VALIDATION ---
    model.eval()
    val_loss_epoch = 0.0
    
    all_rul_preds, all_rul_trues = [], []
    all_alert_preds, all_alert_trues = [], []
    
    with torch.no_grad():
        for X_batch, rul_batch, alert_batch in val_loader:
            X_batch = X_batch.to(device)
            rul_batch = rul_batch.to(device)
            alert_batch = alert_batch.to(device)
            
            rul_pred, alert_pred = model(X_batch)
            total_loss, mse, bce = model_module.combined_loss(
                rul_pred, rul_batch, alert_pred, alert_batch, alpha=ALPHA, beta=BETA)
            
            val_loss_epoch += total_loss.item() * X_batch.size(0)
            
            all_rul_preds.extend(rul_pred.cpu().numpy())
            all_rul_trues.extend(rul_batch.cpu().numpy())
            all_alert_preds.extend(alert_pred.cpu().numpy())
            all_alert_trues.extend(alert_batch.cpu().numpy())
            
    val_loss_epoch /= len(val_loader.dataset)
    
    # Calculate Metrics
    rmse = np.sqrt(mean_squared_error(all_rul_trues, all_rul_preds))
    
    # Threshold alert probabilities at 0.5 for F1 score
    alert_preds_binary = (np.array(all_alert_preds) >= 0.5).astype(int)
    
    # Skip AUC if only one class exists in validation batch (can happen in extreme cases)
    try:
        auc = roc_auc_score(all_alert_trues, all_alert_preds)
    except ValueError:
        auc = 0.5
        
    f1 = f1_score(all_alert_trues, alert_preds_binary, zero_division=0)
    
    # Logging
    history['train_loss'].append(train_loss_epoch)
    history['val_loss'].append(val_loss_epoch)
    history['val_rmse'].append(rmse)
    history['val_f1'].append(f1)
    history['val_auc'].append(auc)
    
    print(f"Epoch {epoch+1:03d}/{EPOCHS} | "
          f"Train Loss: {train_loss_epoch:.2f} | "
          f"Val Loss: {val_loss_epoch:.2f} | "
          f"Val RMSE: {rmse:.2f} | Val F1: {f1:.3f} | Val AUC: {auc:.3f}")
    
    # Scheduler step
    scheduler.step(val_loss_epoch)
    
    # Early Stopping Task
    if val_loss_epoch < best_val_loss:
        best_val_loss = val_loss_epoch
        best_model_weights = copy.deepcopy(model.state_dict())
        es_counter = 0
        # Save best model
        torch.save(best_model_weights, 'best_multitask_model.pth')
    else:
        es_counter += 1
        if es_counter >= ES_PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch+1}! Restoring best weights...")
            model.load_state_dict(best_model_weights)
            break

print("\nTraining Complete.")
print(f"Best Validation Loss: {best_val_loss:.2f}")

# Plotting Loss Curves
plt.figure(figsize=(10, 5))
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (α*MSE + β*BCE)')
plt.legend()
plt.grid(True)
plt.savefig('learning_curves.png')
plt.close()

print("Saved 'learning_curves.png' and 'best_multitask_model.pth'.")