import pandas as pd
import numpy as np
from .paths import DATA_PROCESSED_DIR, DATA_RAW_DIR

files = ['FD001', 'FD002', 'FD003', 'FD004']

RUL_CAP = 125
ALERT_THRESHOLD = 30

print("Starting Step 3: Generate RUL & Alert Labels...\n")

for fd in files:
    print(f"--- Processing {fd} ---")
    
    # Load preprocessed datasets
    train_df = pd.read_csv(DATA_PROCESSED_DIR / f'train_{fd}_preprocessed.csv')
    test_df = pd.read_csv(DATA_PROCESSED_DIR / f'test_{fd}_preprocessed.csv')
    
    # Load true RUL values for test set
    true_rul = pd.read_csv(DATA_RAW_DIR / f'RUL_{fd}.txt', sep=r'\s+', header=None, names=['RUL'])
    # true_rul corresponds to the RUL at the END of each unit's trajectory in the test set.
    
    # --- Process TRAIN set ---
    # Group by unit to get the max cycle
    train_max_cycles = train_df.groupby('unit')['cycle'].max().reset_index()
    train_max_cycles.rename(columns={'cycle': 'max_cycle'}, inplace=True)
    train_df = train_df.merge(train_max_cycles, on=['unit'], how='left')
    
    # Task 1: RUL = max_cycle_per_engine - current_cycle
    train_df['RUL'] = train_df['max_cycle'] - train_df['cycle']
    
    # Task 2: Cap RUL at 125 (piecewise linear RUL)
    train_df['RUL'] = train_df['RUL'].clip(upper=RUL_CAP)
    
    # Task 3: Binary alert (classification): alert = 1 if RUL <= 30 cycles, else 0
    train_df['alert'] = np.where(train_df['RUL'] <= ALERT_THRESHOLD, 1, 0)
    train_df.drop(columns=['max_cycle'], inplace=True)
    
    # --- Process TEST set ---
    # For the test set, the engine fails SOME cycles AFTER the last recorded cycle.
    # Total cycles until failure = max cycle in test + true RUL value for that engine
    test_max_cycles = test_df.groupby('unit')['cycle'].max().reset_index()
    test_max_cycles.rename(columns={'cycle': 'max_cycle'}, inplace=True)
    
    # Add the true RUL to get the theoretical end of life
    # Note: true_rul is typically sorted by unit
    test_max_cycles['failure_cycle'] = test_max_cycles['max_cycle'] + true_rul['RUL']
    
    test_df = test_df.merge(test_max_cycles[['unit', 'failure_cycle']], on=['unit'], how='left')
    
    # RUL = theoretical failure cycle - current_cycle
    test_df['RUL'] = test_df['failure_cycle'] - test_df['cycle']
    
    # Cap RUL
    test_df['RUL'] = test_df['RUL'].clip(upper=RUL_CAP)
    
    # Binary alert
    test_df['alert'] = np.where(test_df['RUL'] <= ALERT_THRESHOLD, 1, 0)
    test_df.drop(columns=['failure_cycle'], inplace=True)
    
    # Task 4: Verify class balance for binary labels
    train_positives = train_df['alert'].mean() * 100
    test_positives = test_df['alert'].mean() * 100
    print(f"Class Balance (Train): {train_positives:.2f}% positive (alert=1)")
    print(f"Class Balance (Test):  {test_positives:.2f}% positive (alert=1)")
    
    # Save back
    train_df.to_csv(DATA_PROCESSED_DIR / f'train_{fd}_labeled.csv', index=False)
    test_df.to_csv(DATA_PROCESSED_DIR / f'test_{fd}_labeled.csv', index=False)
    
print("\nOptional Task 5 note: To handle the ~15-20% imbalance (as shown above), models in later steps should use class_weight='balanced' or SMOTE.")
print("Step 3 Complete! Datasets with RUL and alert labels are saved.")
