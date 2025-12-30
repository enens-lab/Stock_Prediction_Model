import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import mixed_precision
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import glob
import os
import gc

# --- CONFIG ---
CONFIG = {
    "DATA_DIR": Path("TrainingData/indicators_data/processed/stocksData"),
    "FORECAST_DIR": Path("forecasts"),
    "MODEL_PATH": Path("lstm_model.h5"),
    "SEQ_LEN": 60,
    "PRED_HORIZON": 5,
    "BATCH_SIZE": 16384,  # MASSIVE batch size for A100 speed
    "EPOCHS": 50,
}
os.makedirs(CONFIG["FORECAST_DIR"], exist_ok=True)

# Enable Mixed Precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

print("Loading Data...")
all_files = glob.glob(str(CONFIG["DATA_DIR"] / "*.csv"))
if not all_files:
    raise ValueError("No data found! Run processor.py first.")

# --- 1. DATA LOADER ---
features_list = []
labels_list = []

# Step Size 1 = Max Data
STEP_SIZE = 1
print(f"Processing {len(all_files)} stocks (Step Size = {STEP_SIZE})...")

for idx, f in enumerate(all_files):
    try:
        df = pd.read_csv(f)
        cols_to_drop = ['date', 'target_5d', 'open', 'high', 'low', 'close', 'volume']
        feature_cols = [c for c in df.columns if c not in cols_to_drop]

        # Sanitization
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
        df.dropna(subset=feature_cols, inplace=True)
        
        if len(df) < CONFIG["SEQ_LEN"] + CONFIG["PRED_HORIZON"]:
            continue

        scaler = StandardScaler()
        # Direct float16 conversion to save RAM
        X_raw = scaler.fit_transform(df[feature_cols].values).astype(np.float16)
        
        future_close = df['close'].shift(-CONFIG["PRED_HORIZON"])
        current_close = df['close']
        returns = (future_close - current_close) / current_close
        targets = (returns > 0.02).astype(int).values 
        
        limit = len(X_raw) - CONFIG["PRED_HORIZON"]
        
        # Fast append
        for i in range(CONFIG["SEQ_LEN"], limit, STEP_SIZE):
            features_list.append(X_raw[i-CONFIG["SEQ_LEN"]:i])
            labels_list.append(targets[i])
            
    except Exception:
        continue
        
    if idx % 200 == 0:
        print(f"   Processed {idx}/{len(all_files)} stocks...")
        gc.collect()

if not features_list:
    raise ValueError("No valid data found!")

print("Converting to NumPy Arrays...")
X = np.array(features_list, dtype=np.float16)
y = np.array(labels_list, dtype=np.int8)

# Clear list to free RAM
del features_list
del labels_list
gc.collect()

print(f"Data Loaded. Shape: {X.shape}")
print(f"   RAM Usage: ~{X.nbytes / 1e9:.2f} GB")
buy_count = np.sum(y, dtype=np.int64)
print(f"   Buy Signals: {buy_count} ({buy_count/len(y):.2%})")

# --- 2. FAST PIPELINE (THE SPEED FIX) ---
print("⚙️ Building High-Performance Pipeline...")

indices = np.arange(len(X))
np.random.shuffle(indices) # Shuffle ONCE globally (Fastest)

val_split = int(len(X) * 0.1)
train_indices = indices[:-val_split]
val_indices = indices[-val_split:]

def create_fast_generator(indices_subset, batch_size):
    # This generator yields BATCHES, not items. 
    # Reduces Python overhead by 16,000x.
    def gen():
        total = len(indices_subset)
        for i in range(0, total, batch_size):
            batch_idxs = indices_subset[i : i + batch_size]
            # Slicing numpy array by list of indices is fast
            yield X[batch_idxs], y[batch_idxs]
            
    return gen

# Create Datasets
train_gen = create_fast_generator(train_indices, CONFIG["BATCH_SIZE"])
val_gen = create_fast_generator(val_indices, CONFIG["BATCH_SIZE"])

# TF Dataset Definition
output_sig = (
    tf.TensorSpec(shape=(None, CONFIG["SEQ_LEN"], X.shape[2]), dtype=tf.float16),
    tf.TensorSpec(shape=(None,), dtype=tf.int8)
)

train_ds = tf.data.Dataset.from_generator(train_gen, output_signature=output_sig)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE) # GPU Pre-fetching

val_ds = tf.data.Dataset.from_generator(val_gen, output_signature=output_sig)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# --- 3. BUILD MODEL ---
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(CONFIG["SEQ_LEN"], X.shape[2])),
    BatchNormalization(),
    LSTM(128, return_sequences=True),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid', dtype='float32') 
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'AUC'])

# --- 4. TRAIN ---
pos = buy_count
neg = len(y) - pos
total = len(y)
weight_for_0 = (1 / neg) * (total / 2.0)
weight_for_1 = (1 / pos) * (total / 2.0)
class_weight = {0: weight_for_0, 1: weight_for_1}

print("Training Model (Turbo Mode)...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=CONFIG["EPOCHS"],
    callbacks=[
        EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(patience=3)
    ],
    class_weight=class_weight
)

model.save(CONFIG["MODEL_PATH"])
print("Model Saved!")

# --- 5. FORECASTING (Historical Window) ---
print("Generating Historical Forecasts (Last 365 Days)...")

# Free RAM
del X, y, train_ds, val_ds, indices
gc.collect()

BACKTEST_DAYS = 365 

for idx, f in enumerate(all_files):
    try:
        df = pd.read_csv(f)
        feature_cols = [c for c in df.columns if c not in cols_to_drop]
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
        df.dropna(subset=feature_cols, inplace=True)

        if len(df) < CONFIG["SEQ_LEN"] + 10: continue
        
        ticker = Path(f).stem
        scaler = StandardScaler()
        X_raw = scaler.fit_transform(df[feature_cols].values)
        
        start_idx = max(CONFIG["SEQ_LEN"], len(X_raw) - BACKTEST_DAYS)
        batch_seqs = []
        batch_dates = []
        batch_closes = []
        
        for i in range(start_idx, len(X_raw)):
            seq = X_raw[i-CONFIG["SEQ_LEN"]:i]
            batch_seqs.append(seq)
            batch_dates.append(df['date'].iloc[i])
            batch_closes.append(df['close'].iloc[i])
            
        if not batch_seqs: continue
            
        batch_seqs = np.array(batch_seqs)
        probs = model.predict(batch_seqs, verbose=0).flatten()
        
        out_df = pd.DataFrame({
            'date': batch_dates,
            'ticker': ticker,
            'prob_1w': probs, 
            'close': batch_closes
        })
        
        out_df.to_csv(CONFIG["FORECAST_DIR"] / f"{ticker}_forecast.csv", index=False)
        
    except Exception:
        continue
    
    if idx % 200 == 0:
        print(f"   Forecasted history for {idx}/{len(all_files)} stocks...")

print("Historical forecasts generated.")