import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .paths import DATA_PROCESSED_DIR

dataset_name = 'FD001' # Let's demonstrate with FD001. User can loop over others if desired.

W = 30 # Window size

print(f"Starting Step 4: Build Sliding Window Sequences for {dataset_name}...\n")

# Load the labeled data
train_df = pd.read_csv(DATA_PROCESSED_DIR / f'train_{dataset_name}_labeled.csv')
test_df = pd.read_csv(DATA_PROCESSED_DIR / f'test_{dataset_name}_labeled.csv')

# Feature columns (exclude unit, cycle, RUL, alert)
feature_cols = [c for c in train_df.columns if c not in ['unit', 'cycle', 'RUL', 'alert']]

# Function to create sequences per engine
def create_sequences(df, window_size, feature_cols):
    X, y_rul, y_alert = [], [], []
    for unit_id, group in df.groupby('unit'):
        # convert to numpy
        data = group[feature_cols].values
        rul = group['RUL'].values
        alert = group['alert'].values
        
        # If the engine series is shorter than the window, we skip or handle it. 
        # For CMAPSS, train trajectories are usually > 30, test might be shorter sometimes.
        # Simple approach: skip if less than window size for now, or pad.
        # In test set, we often take the LAST sequence only, but let's build standard sliding windows.
        if len(data) >= window_size:
            for i in range(len(data) - window_size + 1):
                X.append(data[i : i + window_size, :])
                y_rul.append(rul[i + window_size - 1])
                y_alert.append(alert[i + window_size - 1])
                
    return np.array(X), np.array(y_rul), np.array(y_alert)

# Task: Split 80% train engines / 20% val engines
unique_engines = train_df['unit'].unique()
np.random.seed(42)
np.random.shuffle(unique_engines)

split_idx = int(len(unique_engines) * 0.8)
train_engines = unique_engines[:split_idx]
val_engines = unique_engines[split_idx:]

train_split_df = train_df[train_df['unit'].isin(train_engines)]
val_split_df = train_df[train_df['unit'].isin(val_engines)]

print(f"Split by Engine ID:")
print(f"Train Engines ({len(train_engines)}): {train_engines[:5]}...")
print(f"Val Engines ({len(val_engines)}): {val_engines[:5]}...\n")

# Build sliding windows
X_train, y_rul_train, y_alert_train = create_sequences(train_split_df, W, feature_cols)
X_val, y_rul_val, y_alert_val = create_sequences(val_split_df, W, feature_cols)

# We can optionally also build test sequences, usually just the last sequence per engine is evaluated for RUL
# but let's just make full sequences for now.
X_test, y_rul_test, y_alert_test = create_sequences(test_df, W, feature_cols)

# PyTorch Dataset
class CMAPSSDataset(Dataset):
    def __init__(self, X, y_rul, y_alert):
        # Convert to tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_rul = torch.tensor(y_rul, dtype=torch.float32)
        self.y_alert = torch.tensor(y_alert, dtype=torch.float32) # Using float for BCEWithLogitsLoss or CrossEntropyLoss if long
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y_rul[idx], self.y_alert[idx]

train_dataset = CMAPSSDataset(X_train, y_rul_train, y_alert_train)
val_dataset = CMAPSSDataset(X_val, y_rul_val, y_alert_val)
test_dataset = CMAPSSDataset(X_test, y_rul_test, y_alert_test)

# DataLoaders (batch_size=64 for demonstration)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print("Created DataLoaders:")
print(f"Train batches: {len(train_loader)}")
print(f"Val batches:   {len(val_loader)}")
print(f"Test batches:  {len(test_loader)}\n")

# Fetch one batch to verify
for x_batch, rul_batch, alert_batch in train_loader:
    print("Example Batch Shapes:")
    print(f"X (Features):   {x_batch.shape} --> (Batch, Window, Features)")
    print(f"y (RUL):        {rul_batch.shape} --> (Batch)")
    print(f"y (Alert):      {alert_batch.shape} --> (Batch)")
    
    # Save the custom Dataset class definition to a reusable module format if needed
    break

print("\nStep 4 Complete! PyTorch Datasets and DataLoaders are implemented and verified.")
