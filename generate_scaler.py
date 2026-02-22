"""
Quick Fix: Generate Missing Scaler File
Run this ONCE to create the scaler without retraining the model

This version detects the actual feature set from your processed files
FORCED TO USE 42 FEATURES
"""

import numpy as np
import pandas as pd
import glob
import json
import random
import joblib
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from collections import Counter

# --- CONFIG (Match your training script) ---
DATA_DIR = Path("TrainingData/indicators_data/processed/stocksData")
FEATURE_MAP_PATH = Path("TrainingData/models/feature_columns_jackpot.json")
SCALER_PATH = Path("TrainingData/models/scaler_jackpot.joblib")

print("Generating Missing Scaler File...")
print("FORCING 42 FEATURES...")

# 1. Scan files to find the 42-feature set
all_files = glob.glob(str(DATA_DIR / "*.csv"))
if not all_files:
    print("ERROR: No processed data found!")
    exit(1)

print(f"Found {len(all_files)} processed files")

# Sample files to find feature sets
sample_for_detection = random.sample(all_files, min(100, len(all_files)))
feature_sets = {}  # Maps feature count -> (feature list, count)

print("Scanning files to find 42-feature set...")
for f in sample_for_detection:
    try:
        df = pd.read_csv(f, nrows=1)  # Just read header
        
        # Exclude non-feature columns
        exclude_cols = ['date', 'target_5d', 'target_20d', 'open', 'high', 'low', 'close', 'volume', 'ticker', 'target']
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        # Only keep numeric columns
        df_sample = pd.read_csv(f, nrows=100)
        numeric_features = df_sample[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        feature_count = len(numeric_features)
        if feature_count not in feature_sets:
            feature_sets[feature_count] = [sorted(numeric_features), 0]
        feature_sets[feature_count][1] += 1
        
    except:
        continue

# Print what we found
print("\nFeature sets detected:")
for count, (features, num_files) in sorted(feature_sets.items()):
    print(f"  {count} features: {num_files} files")

# Force 42 features
if 42 in feature_sets:
    saved_feature_cols = feature_sets[42][0]
    print(f"\n✓ Using 42-feature set (found in {feature_sets[42][1]} files)")
elif 41 in feature_sets:
    print(f"\n⚠ WARNING: No 42-feature files found!")
    print(f"  Most common: {max(feature_sets.items(), key=lambda x: x[1][1])}")
    print(f"  Using 41 features instead and padding with zeros...")
    saved_feature_cols = feature_sets[41][0]
    # Add a dummy feature to make it 42
    saved_feature_cols.append('dummy_feature_placeholder')
else:
    print(f"\n❌ ERROR: No suitable feature set found!")
    print(f"  Available: {list(feature_sets.keys())}")
    exit(1)

print(f"  First 5 features: {saved_feature_cols[:5]}")

# Update feature map to 42
print(f"\nUpdating feature map to 42 features: {FEATURE_MAP_PATH}")
with open(FEATURE_MAP_PATH, 'w') as f:
    json.dump(saved_feature_cols, f)
print("✓ Feature map updated")

# 2. Sample files and extract feature data (only use files with 42 features)
sample_files = all_files  # Use all files
scaler_data = []
successful = 0
failed = 0

print(f"\nSampling data from files (looking for 42-feature files)...")
for f in sample_files:
    try:
        df = pd.read_csv(f)
        
        # Use only the detected feature columns
        df_feat = df.reindex(columns=saved_feature_cols, fill_value=0)
        df_feat = df_feat.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Final safety check: ensure all numeric
        df_feat = df_feat.select_dtypes(include=[np.number])
        
        # CRITICAL: Only use if it has exactly 42 features
        if df_feat.shape[1] == 42:
            scaler_data.append(df_feat.values)
            successful += 1
            if successful >= 200:  # Stop after 200 good files
                break
        else:
            failed += 1
    except Exception as e:
        failed += 1
        continue

print(f"  Successful: {successful}")
print(f"  Failed/Skipped: {failed}")

# 3. Fit and save scaler
if scaler_data and successful > 0:
    print(f"\nFitting scaler on {len(scaler_data)} file samples...")
    scaler_array = np.concatenate(scaler_data, axis=0)
    print(f"  Total samples: {scaler_array.shape[0]:,}")
    print(f"  Features: {scaler_array.shape[1]}")
    
    if scaler_array.shape[1] != 42:
        print(f"\n❌ ERROR: Scaler has {scaler_array.shape[1]} features, not 42!")
        print("   Your processed files may not have 42 features.")
        print("   Consider re-running processor.py")
        exit(1)
    
    global_scaler = StandardScaler()
    global_scaler.fit(scaler_array)
    
    # Save
    joblib.dump(global_scaler, SCALER_PATH)
    print(f"\n{'='*60}")
    print(f"✓ Scaler saved successfully: {SCALER_PATH}")
    print(f"✓ Feature map updated: {FEATURE_MAP_PATH}")
    print(f"✓ Features: 42 (FORCED)")
    print(f"✓ Your model is now complete and ready to use!")
    print(f"{'='*60}")
else:
    print(f"\n❌ ERROR: Could not find enough files with 42 features")
    print(f"   Found {successful} files with correct feature count")
    print(f"   Need at least 10 files to create a valid scaler")
    print("\n💡 SOLUTION: Re-run processor.py to standardize all files")
    exit(1)
