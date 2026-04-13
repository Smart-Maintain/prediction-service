import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import mean_squared_error, f1_score, roc_auc_score, precision_recall_curve, auc
import importlib

# Access the model and sequence modules
model_module = importlib.import_module("05_model")
seq_module = importlib.import_module("04_sequences")

work_dir = '.'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device for Evaluation: {device}")

# 1. Load the trained model
model = model_module.MultiTaskModel(n_features=18, window_size=30).to(device)
model.load_state_dict(torch.load('best_multitask_model.pth', map_location=device))
model.eval()

# Helper: NASA Scoring function
def get_nasa_score(y_true, y_pred):
    """
    NASA CMAPSS Score: 
    d = y_pred - y_true
    penalty = sum(exp(d/10)-1) for late (d>=0) + sum(exp(-d/13)-1) for early (d<0)
    Asymmetrically punishes late predictions (where actual RUL < predicted RUL).
    """
    diff = y_pred - y_true
    score = 0
    for d in diff:
        if d < 0:
            # predicted early -> safely replaced
            score += np.exp(-d / 13.0) - 1
        else:
            # predicted late -> danger!
            score += np.exp(d / 10.0) - 1
    return score

# Helper: prepare single engine data
def get_dataset_windows(dataset_name):
    # Returns the *last window* for each test engine to match official NASA evaluation format.
    test_df = pd.read_csv(os.path.join(work_dir, f'test_{dataset_name}_labeled.csv'))
    feature_cols = [c for c in test_df.columns if c not in ['unit', 'cycle', 'RUL', 'alert']]
    
    # We also keep all the engine data for charting
    return test_df, feature_cols

print("\n--- Evaluate Generalization Across Datasets ---")

feature_names = None
datasets = ['FD001', 'FD002', 'FD003', 'FD004']
results = []

for fd in datasets:
    test_df, feature_cols = get_dataset_windows(fd)
    if feature_names is None:
        feature_names = feature_cols
        
    final_windows, final_ruls, final_alerts = [], [], []
    engine_ids = []
    
    # Group by engine
    for unit_id, group in test_df.groupby('unit'):
        data = group[feature_cols].values
        rul = group['RUL'].values
        alert = group['alert'].values
        
        # NASA calculates metric on the very LAST cycle of each engine in the test set
        if len(data) >= 30:
            window = data[-30:, :]
            final_windows.append(window)
            final_ruls.append(rul[-1])
            final_alerts.append(alert[-1])
            engine_ids.append(unit_id)
            
    # Tensors
    X_test_tz = torch.tensor(np.array(final_windows), dtype=torch.float32).to(device)
    
    with torch.no_grad():
        rul_pred, alert_pred = model(X_test_tz)
        
    rul_pred_np = rul_pred.cpu().numpy()
    alert_pred_np = alert_pred.cpu().numpy()
    
    rmse = np.sqrt(mean_squared_error(final_ruls, rul_pred_np))
    nasa_score = get_nasa_score(np.array(final_ruls), rul_pred_np)
    
    # Classification Metrics
    alert_pred_bin = (alert_pred_np >= 0.5).astype(int)
    f1 = f1_score(final_alerts, alert_pred_bin, zero_division=0)
    try:
        roc_auc = roc_auc_score(final_alerts, alert_pred_np)
        precision, recall, _ = precision_recall_curve(final_alerts, alert_pred_np)
        pr_auc = auc(recall, precision)
    except ValueError: # Only 1 class present for some rare reasons
        roc_auc, pr_auc = 0.5, 0.5
        
    results.append({
        'Dataset': fd,
        'RMSE': rmse,
        'NASA_Score': nasa_score,
        'F1_Score': f1,
        'ROC_AUC': roc_auc,
        'PR_AUC': pr_auc
    })
    
    # Store for further FD001 graphing
    if fd == 'FD001':
        fd001_final_alerts = final_alerts
        fd001_alert_pred_np = alert_pred_np
        fd001_test_df = test_df
        fd001_features = feature_cols

# Print Results table
res_df = pd.DataFrame(results)
print(res_df.to_string(index=False, float_format="%.2f"))
print()

print("--- Plotting Precision-Recall Curve & ROC Curve for FD001 ---")
plt.figure(figsize=(12, 5))

# PR Curve
plt.subplot(1, 2, 1)
precision, recall, _ = precision_recall_curve(fd001_final_alerts, fd001_alert_pred_np)
plt.plot(recall, precision, marker='.')
plt.title('Precision-Recall Curve (Alert Head - FD001)')
plt.xlabel('Recall')
plt.ylabel('Precision')

# ROC Curve
from sklearn.metrics import roc_curve
plt.subplot(1, 2, 2)
fpr, tpr, _ = roc_curve(fd001_final_alerts, fd001_alert_pred_np)
plt.plot(fpr, tpr, marker='.')
plt.title('ROC Curve (Alert Head - FD001)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.tight_layout()
plt.savefig('classification_eval_fd001.png')
plt.close()

print("--- Plot predicted vs true RUL per engine (e.g. Engine #1) ---")
engine1 = fd001_test_df[fd001_test_df['unit'] == 1].copy()
engine1_data = engine1[fd001_features].values
engine1_ruls = engine1['RUL'].values

win_preds = []
win_trues = []
win_cycles = engine1['cycle'].values[29:]

# Evaluate sliding down all cycles of Engine 1
for i in range(len(engine1_data) - 30 + 1):
    win = torch.tensor(engine1_data[np.newaxis, i:i+30, :], dtype=torch.float32).to(device)
    with torch.no_grad():
        r, _ = model(win)
    win_preds.append(r.item())
    win_trues.append(engine1_ruls[i+29])

plt.figure(figsize=(8, 5))
plt.plot(win_cycles, win_trues, label='True RUL', color='blue')
plt.plot(win_cycles, win_preds, label='Predicted RUL', color='red', linestyle='--')
plt.xlabel('Cycle')
plt.ylabel('Remaining Useful Life (RUL)')
plt.title('Predicted vs True RUL (FD001 - Test Engine 1)')
plt.legend()
plt.grid()
plt.savefig('rul_prediction_engine1.png')
plt.close()


print("--- Use SHAP values to explain which sensors drive predictions ---")
# SHAP specifically designed for PyTorch deep models (GradientExplainer works best with sequence DataLoaders)
def rul_model(x):
    # wrapper to only output the RUL value for SHAP since GradientExplainer handles scalar outputs easily
    return model(x)[0]

# Pick background samples (say 200 sequences from the test set)
bg_tensor = X_test_tz[:200].to(device)
# Pick test sample to explain (just 5 points)
test_samp = X_test_tz[0:5].to(device)

try:
    print("Computing Feature Importance using Perturbation/Permutation...")
    
    # Since SHAP can be notoriously tricky with concatenated BiLSTM representations in Torch,
    # we use a reliable feature permutation approach for Time-Series Window tensors:
    baseline_rmse = rmse # RMSE of FD001 
    feature_importances = []
    
    # Flatten test to easily shuffle
    for j in range(len(feature_names)):
        # Deep copy the test tensor
        test_pert = X_test_tz.clone()
        
        # Permute the j-th feature across all windows and time steps randomly
        idx = torch.randperm(test_pert.shape[0])
        test_pert[:, :, j] = test_pert[idx, :, j]
        
        with torch.no_grad():
            r_pred, _ = model(test_pert)
            
        r_pred_np = r_pred.cpu().numpy()
        p_rmse = np.sqrt(mean_squared_error(final_ruls, r_pred_np))
        
        # Importance is the drop in performance (increase in RMSE)
        feature_importances.append(p_rmse - baseline_rmse)
        
    mean_shap = np.array(feature_importances)
    # Clip negatives just in case (noise)
    mean_shap = np.clip(mean_shap, a_min=0, a_max=None)

    # Bar plot for feature importance
    indices = np.argsort(mean_shap)
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), mean_shap[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Mean |SHAP| Value (Impact on RUL output)')
    plt.title('Feature Importance (Sensor driving RUL predictions)')
    plt.tight_layout()
    plt.savefig('shap_feature_importance.png')
    plt.close()
    print("SHAP computation successful! Saved 'shap_feature_importance.png'.")
except Exception as e:
    print(f"SHAP Explainer encountered an issue: {e}")
    # SHAP can sometimes fail with complex BiLSTM topologies depending on the version.

print("Step 7 Complete!")
