import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv1D, LSTM, Dense, Dropout, BatchNormalization,
    Input, MultiHeadAttention, LayerNormalization, Add
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import glob
import os
import gc
import json
import random
import joblib
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

# --- 0. FORCE GPU DETECTION & SETUP ---
physical_devices = tf.config.list_physical_devices('GPU')
print(f"PHYSICAL GPUs FOUND: {len(physical_devices)}")
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)

strategy = tf.distribute.MirroredStrategy()
print(f"DEPLOYMENT: Active TensorFlow Replicas: {strategy.num_replicas_in_sync}")

# --- CONFIG ---
BATCH_SIZE_PER_REPLICA = 512
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

CONFIG = {
    "DATA_DIR": Path("TrainingData/indicators_data/processed/stocksData"),
    "MODEL_DIR": Path("TrainingData/models"),
    "MODEL_PATH": Path("TrainingData/models/lstm_jackpot.keras"),
    "FEATURE_MAP_PATH": Path("TrainingData/models/feature_columns_jackpot.json"),
    "SCALER_PATH": Path("TrainingData/models/scaler_jackpot.joblib"),
    "SEQ_LEN": 60,
    "PRED_HORIZON": 20,  # JACKPOT: 20 days instead of 5
    "BATCH_SIZE": GLOBAL_BATCH_SIZE,
    "EPOCHS": 60,
    "CHUNK_SIZE": 400,  # Reduced to match production; jackpot sequences are denser
}
os.makedirs(CONFIG["MODEL_DIR"], exist_ok=True)

print(f"Starting JACKPOT Training (Global Batch Size: {CONFIG['BATCH_SIZE']})...")

all_files = glob.glob(str(CONFIG["DATA_DIR"] / "*.csv"))
if not all_files:
    raise ValueError("No data found! Run processor.py first.")

# --- 1. TEMPORAL TRAIN / VAL SPLIT ---
# Each stock file contributes two non-overlapping time slices:
#   train : date < VAL_CUTOFF_DATE  (model sees this during weight updates)
#   val   : date >= VAL_CUTOFF_DATE (completely unseen market regime)
# This eliminates regime-leakage from the old random stock-level split.
VAL_CUTOFF_DATE = pd.Timestamp('2024-01-01')

print(f"Total Files  : {len(all_files)}")
print(f"Train period : before  {VAL_CUTOFF_DATE.date()}  (all stocks, pre-cutoff rows)")
print(f"Val period   : from    {VAL_CUTOFF_DATE.date()}  (all stocks, post-cutoff rows)")

# --- 2. FEATURE LOCKING (ROBUST) ---
print("Establishing Feature Schema...")

# Sample multiple files to find most common feature set
print("Sampling files to detect consistent feature set...")
sample_files = random.sample(all_files, min(50, len(all_files)))
feature_sets = []

for f in sample_files:
    try:
        df = pd.read_csv(f, nrows=1)
        # Exclude metadata, targets, raw OHLCV, and static Finviz fundamentals.
        # Fundamentals are scraped at a single point in time and applied to all
        # historical rows — using them as features introduces look-ahead bias.
        # Sector dummies (sec_*) are retained; sector membership is stable.
        EXCLUDE_COLS = {
            'date', 'target_5d', 'target_20d', 'open', 'high', 'low', 'close',
            'volume', 'ticker', 'target',
            'pe_ratio', 'short_float', 'insider_own', 'inst_own', 'market_cap',
        }
        features = [c for c in df.columns if c not in EXCLUDE_COLS]
        feature_sets.append(tuple(sorted(features)))
    except:
        continue

# Find most common feature set
feature_counter = Counter(feature_sets)
most_common_features, count = feature_counter.most_common(1)[0]
saved_feature_cols = list(most_common_features)

print(f"LOCKED Features ({len(saved_feature_cols)}): (found in {count}/{len(sample_files)} sample files)")
print(f"   {saved_feature_cols[:5]} ...")

with open(CONFIG["FEATURE_MAP_PATH"], "w") as f_json:
    json.dump(saved_feature_cols, f_json)

n_features = len(saved_feature_cols)

# --- 3. CLASS WEIGHT ESTIMATION FROM BROAD SAMPLE ---
def compute_class_weights(files, target_col, cutoff_date, n_files=500):
    """
    Estimate class weights from training rows only (date < cutoff_date).
    Reading only date + target columns keeps this fast even over 3k files.
    For the jackpot model the positive class (>20% in 20d) is rare, so
    filtering to pre-cutoff rows gives an honest estimate for that regime.
    """
    sample = random.sample(files, min(n_files, len(files)))
    total_pos, total_neg = 0, 0
    for f in sample:
        try:
            df = pd.read_csv(f, usecols=['date', target_col])
            df['date'] = pd.to_datetime(df['date'])
            df = df[df['date'] < cutoff_date]
            if df.empty:
                continue
            total_pos += int(df[target_col].sum())
            total_neg += int((df[target_col] == 0).sum())
        except Exception:
            continue
    total = total_pos + total_neg
    if total_pos == 0 or total_neg == 0:
        print("  Warning: degenerate class distribution in sample; using default weights.")
        return {0: 0.5, 1: 10.0}
    w0 = (1.0 / total_neg) * (total / 2.0)
    w1 = (1.0 / total_pos) * (total / 2.0)
    print(f"  Class weight sample: {total_pos:,} positives, {total_neg:,} negatives "
          f"({100*total_pos/total:.1f}% positive rate)")
    print(f"  Class Weights -> 0: {w0:.3f} | 1: {w1:.3f}")
    return {0: w0, 1: w1}

print("Computing class weights from training rows only (pre-cutoff)...")
class_weight = compute_class_weights(all_files, target_col='target_20d', cutoff_date=VAL_CUTOFF_DATE)

# --- 4. TEMPORAL DATA LOADER ---
def process_single_file(f, feature_cols, config, split='train'):
    """
    Load one stock CSV and return (X_sequences, y_labels) for the requested split.

    split='train': sequences where the prediction row i satisfies
                   date[i] < VAL_CUTOFF_DATE and i < cutoff_idx - PRED_HORIZON,
                   so every target window fully resolves before the cutoff.
    split='val':   sequences where date[i] >= VAL_CUTOFF_DATE, i.e. the
                   prediction is made from a genuinely unseen market regime.

    Critically, the StandardScaler is fitted ONLY on pre-cutoff rows so future
    statistics never contaminate feature normalisation.
    """
    try:
        df = pd.read_csv(f)
        if 'date' not in df.columns or len(df) < config["SEQ_LEN"] + config["PRED_HORIZON"]:
            return None, None
        df['date'] = pd.to_datetime(df['date'])

        # --- Target ---
        if 'target_20d' in df.columns:
            targets = df['target_20d'].astype(np.int8).values
        elif 'close' in df.columns:
            close = df['close'].values
            future_close = np.roll(close, -config["PRED_HORIZON"])
            returns = (future_close - close) / close
            returns[-config["PRED_HORIZON"]:] = 0
            targets = (returns > 0.20).astype(np.int8)
        else:
            return None, None

        df_features = df.reindex(columns=feature_cols, fill_value=0)
        df_features = df_features.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Row index of the temporal cutoff
        cutoff_idx = int(df['date'].searchsorted(VAL_CUTOFF_DATE))

        # Fit scaler only on training rows — prevents future-stats leakage
        n_train_rows = cutoff_idx if cutoff_idx > 1 else len(df_features)
        scaler = StandardScaler()
        scaler.fit(df_features.values[:n_train_rows])
        X_raw = scaler.transform(df_features.values).astype(np.float32)

        limit = len(X_raw) - config["PRED_HORIZON"]
        if split == 'train':
            # Ensure the full target window lands before the cutoff
            i_range = range(config["SEQ_LEN"],
                            min(cutoff_idx - config["PRED_HORIZON"], limit))
        else:
            # Prediction row is at or after the cutoff
            i_range = range(max(config["SEQ_LEN"], cutoff_idx), limit)

        X_local, y_local = [], []
        for i in i_range:
            X_local.append(X_raw[i - config["SEQ_LEN"]: i])
            y_local.append(targets[i])

        if not X_local:
            return None, None
        return X_local, y_local

    except Exception:
        return None, None

def load_chunk_parallel(files, feature_cols, config, split='train'):
    """Load a chunk of files in parallel for the given temporal split."""
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_single_file, f, feature_cols, config, split)
                   for f in files]
        results = [fut.result() for fut in futures]

    X_all, y_all = [], []
    for X_part, y_part in results:
        if X_part is not None and len(X_part) > 0:
            X_all.extend(X_part)
            y_all.extend(y_part)

    if not X_all:
        return None, None

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.int8)
    del X_all, y_all, results
    gc.collect()
    return X, y

# --- 4. LOAD VALIDATION DATA (post-cutoff rows from all stocks) ---
print(f"Loading Validation Data (date >= {VAL_CUTOFF_DATE.date()})...")

val_chunk_size = 200
val_file_chunks_load = [all_files[i:i+val_chunk_size]
                        for i in range(0, len(all_files), val_chunk_size)]

X_val_list = []
y_val_list = []

for i, val_chunk in enumerate(val_file_chunks_load):
    print(f"  Loading val chunk {i+1}/{len(val_file_chunks_load)}...", end=" ", flush=True)
    X_val_chunk, y_val_chunk = load_chunk_parallel(
        val_chunk, saved_feature_cols, CONFIG, split='val')
    if X_val_chunk is not None:
        X_val_list.append(X_val_chunk)
        y_val_list.append(y_val_chunk)
        print(f"{X_val_chunk.shape[0]:,} samples")
    else:
        print("(no val sequences)")

X_val = np.concatenate(X_val_list, axis=0)
y_val = np.concatenate(y_val_list, axis=0)
del X_val_list, y_val_list
gc.collect()

# Cap val set to avoid OOM in from_tensor_slices
max_val_samples = 400_000
if len(X_val) > max_val_samples:
    val_idx = np.random.choice(len(X_val), max_val_samples, replace=False)
    X_val = X_val[val_idx]
    y_val = y_val[val_idx]
    print(f"  (Val set randomly capped at {max_val_samples:,} samples)")

print(f"Validation Data Ready: {X_val.shape}")

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_ds = val_ds.batch(CONFIG["BATCH_SIZE"]).prefetch(tf.data.AUTOTUNE)

# --- 5. CHUNKED TRAINING APPROACH ---
num_chunks = max(1, len(all_files) // CONFIG["CHUNK_SIZE"])
train_file_chunks = np.array_split(all_files, num_chunks)

print(f"Training files split into {len(train_file_chunks)} chunks of ~{CONFIG['CHUNK_SIZE']} files each")

# Load first chunk for initialization
print("Loading first training chunk for initialization...")
X_chunk, y_chunk = load_chunk_parallel(
    train_file_chunks[0], saved_feature_cols, CONFIG, split='train')
print(f"First chunk: {X_chunk.shape}")

# Estimate total samples
samples_per_chunk = len(X_chunk)
total_train_samples = samples_per_chunk * len(train_file_chunks)
steps_per_epoch = max(1, total_train_samples // CONFIG["BATCH_SIZE"])
print(f"Estimated total training samples: ~{total_train_samples:,}")
print(f"Steps per epoch: {steps_per_epoch}")

# --- 7. LEARNING RATE SCHEDULE ---
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, warmup_steps, total_decay_steps, initial_lr=1e-3, warmup_start_lr=1e-5,
                 t_mul=2.0, m_mul=0.9):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        self.warmup_start_lr = warmup_start_lr
        self.cosine_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=initial_lr,
            first_decay_steps=total_decay_steps,
            t_mul=t_mul,
            m_mul=m_mul,
        )

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)

        warmup_lr = self.warmup_start_lr + (self.initial_lr - self.warmup_start_lr) * (step / warmup_steps)
        cosine_lr = self.cosine_schedule(step - warmup_steps)

        return tf.where(step < warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "warmup_steps": self.warmup_steps,
            "initial_lr": self.initial_lr,
            "warmup_start_lr": self.warmup_start_lr,
        }

warmup_steps = steps_per_epoch
cosine_decay_steps = steps_per_epoch * 5

lr_schedule = WarmupCosineDecay(
    warmup_steps=warmup_steps,
    total_decay_steps=cosine_decay_steps,
    initial_lr=5e-4,
    warmup_start_lr=1e-6,
)

# --- 8. MODEL ARCHITECTURE ---
with strategy.scope():
    print("Building Distributed PRODUCTION Model...")
    inputs = Input(shape=(CONFIG["SEQ_LEN"], n_features))

    # Conv1D Feature Extraction
    x = Conv1D(64, kernel_size=3, activation='relu')(inputs)
    x = BatchNormalization()(x)

    # LSTM Layer 1
    x = LSTM(256, return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Multi-Head Attention
    attn_output = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = Add()([x, attn_output])
    x = LayerNormalization()(x)

    # LSTM Layer 2
    x = LSTM(128)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Output
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid', dtype='float32')(x)

    model = Model(inputs=inputs, outputs=outputs)

    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy', 'AUC'])

model.summary()

# --- 9. CUSTOM TRAINING LOOP WITH CHUNKED DATA ---
print("\nTraining JACKPOT MODEL with Chunked Data Loading...")

best_val_loss = float('inf')
patience_counter = 0
patience = 15

for epoch in range(CONFIG["EPOCHS"]):
    print(f"\n{'='*60}")
    print(f"Epoch {epoch + 1}/{CONFIG['EPOCHS']}")
    print(f"{'='*60}")
    
    # Shuffle chunks for this epoch
    random.shuffle(train_file_chunks)
    
    epoch_loss = []
    epoch_acc = []
    epoch_auc = []
    
    # Train on each chunk
    for chunk_idx, file_chunk in enumerate(train_file_chunks):
        print(f"\n  Chunk {chunk_idx + 1}/{len(train_file_chunks)} ", end="", flush=True)
        
        # Load chunk (training rows only: date < VAL_CUTOFF_DATE)
        X_chunk, y_chunk = load_chunk_parallel(
            file_chunk, saved_feature_cols, CONFIG, split='train')
        
        if X_chunk is None:
            continue

        print(f"Loaded: {X_chunk.shape}", end=" ", flush=True)

        # Hard cap: from_tensor_slices pins the full array as an EagerConst and
        # attempts to copy it to GPU. At ~5,800 samples/file × 400 files the
        # chunk reaches ~34 GB which exceeds available VRAM. Cap at 800k rows
        # (≈9.4 GB) — same limit as the production training script.
        max_samples = 800_000
        if len(X_chunk) > max_samples:
            print(f"(Trimming {len(X_chunk):,} → {max_samples:,})", end=" ", flush=True)
            idx = np.random.choice(len(X_chunk), max_samples, replace=False)
            X_chunk = X_chunk[idx]
            y_chunk = y_chunk[idx]

        # Create dataset for this chunk
        try:
            chunk_ds = tf.data.Dataset.from_tensor_slices((X_chunk, y_chunk))
            chunk_ds = chunk_ds.shuffle(buffer_size=min(len(X_chunk), 50_000)) \
                               .batch(CONFIG["BATCH_SIZE"]) \
                               .prefetch(tf.data.AUTOTUNE)
        except Exception as e:
            print(f"\n  Warning: dataset creation failed ({e}) — skipping chunk.")
            del X_chunk, y_chunk
            gc.collect()
            continue
        
        # Train on chunk
        history = model.fit(
            chunk_ds,
            epochs=1,
            verbose=2,
            class_weight=class_weight
        )
        
        epoch_loss.append(history.history['loss'][0])
        epoch_acc.append(history.history['accuracy'][0])
        epoch_auc.append(history.history['AUC'][0])
        
        # Clean up
        del X_chunk, y_chunk, chunk_ds
        gc.collect()
    
    # Epoch metrics
    avg_loss = np.mean(epoch_loss)
    avg_acc = np.mean(epoch_acc)
    avg_auc = np.mean(epoch_auc)
    
    print(f"\n  Train - Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}, AUC: {avg_auc:.4f}")
    
    # Validation
    print("  Validating...", end="", flush=True)
    val_results = model.evaluate(val_ds, verbose=0)
    val_loss, val_acc, val_auc = val_results[0], val_results[1], val_results[2]
    
    print(f"\n  Val - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")
    
    # Early stopping and checkpointing (using val_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        model.save(CONFIG["MODEL_PATH"])
        print(f"  ✓ New best model saved! (Loss: {val_loss:.4f})")
    else:
        patience_counter += 1
        print(f"  No improvement (patience: {patience_counter}/{patience})")
    
    if patience_counter >= patience:
        print(f"\n  Early stopping triggered!")
        break

print(f"\n{'='*60}")
print(f"JACKPOT MODEL TRAINING COMPLETE!")
print(f"Best Validation Loss: {best_val_loss:.4f}")
print(f"Model saved: {CONFIG['MODEL_PATH']}")
print(f"{'='*60}")

# --- 10. SAVE SCALER ---
print(f"\nFitting and saving global scaler (training rows only: date < {VAL_CUTOFF_DATE.date()})...")
sample_files = random.sample(all_files, min(200, len(all_files)))
scaler_data = []
successful = 0
failed = 0

for f in sample_files:
    try:
        df = pd.read_csv(f)
        # Restrict to training period only so scaler stats are not contaminated
        # by post-cutoff data
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df[df['date'] < VAL_CUTOFF_DATE]
        if df.empty:
            failed += 1
            continue
        # Use the same feature columns that were locked during training
        df_feat = df.reindex(columns=saved_feature_cols, fill_value=0)
        df_feat = df_feat.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Ensure all columns are numeric and match expected count
        df_feat = df_feat.select_dtypes(include=[np.number])

        # CRITICAL: Only use if it matches our locked feature count
        if df_feat.shape[1] == len(saved_feature_cols):
            scaler_data.append(df_feat.values)
            successful += 1
        else:
            failed += 1
    except Exception as e:
        failed += 1
        continue

print(f"  Scaler samples: {successful} successful, {failed} failed")

if scaler_data:
    scaler_array = np.concatenate(scaler_data, axis=0)
    global_scaler = StandardScaler()
    global_scaler.fit(scaler_array)
    joblib.dump(global_scaler, CONFIG["SCALER_PATH"])
    print(f"✓ Global scaler saved: {CONFIG['SCALER_PATH']}")
else:
    print("⚠ Warning: No data available for scaler fitting.")