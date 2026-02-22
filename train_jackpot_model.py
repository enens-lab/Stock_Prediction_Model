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
    "PRED_HORIZON": 20,
    "TARGET_GAIN": 0.20,
    "BATCH_SIZE": GLOBAL_BATCH_SIZE,
    "EPOCHS": 60,
    "CHUNK_SIZE": 700,  # REDUCED from 1000 to prevent OOM
}
os.makedirs(CONFIG["MODEL_DIR"], exist_ok=True)

print(f"Starting JACKPOT Training (Global Batch Size: {CONFIG['BATCH_SIZE']})...")

all_files = glob.glob(str(CONFIG["DATA_DIR"] / "*.csv"))
if not all_files:
    raise ValueError("No data found! Run processor.py first.")

# --- 1. STRICT FILE SPLITTING (Prevent Leakage) ---
random.shuffle(all_files)
split_idx = int(len(all_files) * 0.9)
train_files = all_files[:split_idx]
val_files = all_files[split_idx:]

print(f"Total Files: {len(all_files)}")
print(f"   Train Files: {len(train_files)}")
print(f"   Val Files:   {len(val_files)} (Strictly Unseen)")

# --- 2. FEATURE LOCKING ---
print("Establishing Feature Schema...")
first_df = pd.read_csv(train_files[0])
cols_to_drop = ['date', 'target_5d', 'target_20d', 'open', 'high', 'low', 'close', 'volume', 'ticker', 'target']
saved_feature_cols = [c for c in first_df.columns if c not in cols_to_drop]

print(f"LOCKED Features ({len(saved_feature_cols)}):")
print(f"   {saved_feature_cols[:5]} ...")

with open(CONFIG["FEATURE_MAP_PATH"], "w") as f_json:
    json.dump(saved_feature_cols, f_json)

n_features = len(saved_feature_cols)

# --- 3. OPTIMIZED PARALLEL DATA LOADER ---
def process_single_file(f, feature_cols, config):
    try:
        df = pd.read_csv(f)
        
        if len(df) < config["SEQ_LEN"] + config["PRED_HORIZON"]:
            return None, None
        
        # Check if we have the pre-computed target
        if 'target_20d' in df.columns:
            # Use pre-computed target from processor
            targets = df['target_20d'].astype(np.int8).values
        elif 'close' in df.columns:
            # Fallback: compute target on the fly
            close = df['close'].values
            future_close = np.roll(close, -config["PRED_HORIZON"])
            returns = (future_close - close) / close
            returns[-config["PRED_HORIZON"]:] = 0
            targets = (returns > config["TARGET_GAIN"]).astype(np.int8)
        else:
            return None, None
        
        df_features = df.reindex(columns=feature_cols, fill_value=0)
        df_features = df_features.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        scaler = StandardScaler()
        X_raw = scaler.fit_transform(df_features.values).astype(np.float32)
        
        X_local = []
        y_local = []
        limit = len(X_raw) - config["PRED_HORIZON"]
        
        for i in range(config["SEQ_LEN"], limit):
            X_local.append(X_raw[i-config["SEQ_LEN"]:i])
            y_local.append(targets[i])
        
        return X_local, y_local
    except Exception:
        return None, None

def load_chunk_parallel(files, feature_cols, config):
    """Load a chunk of files in parallel"""
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_single_file, f, feature_cols, config) for f in files]
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

# --- 4. LOAD VALIDATION DATA (Small enough to fit in memory) ---
print("Loading Validation Data...")
X_val, y_val = load_chunk_parallel(val_files, saved_feature_cols, CONFIG)
print(f"Validation Data Ready: {X_val.shape}")

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_ds = val_ds.batch(CONFIG["BATCH_SIZE"]).prefetch(tf.data.AUTOTUNE)

# --- 5. CHUNKED TRAINING APPROACH ---
# Split training files into manageable chunks
num_chunks = max(1, len(train_files) // CONFIG["CHUNK_SIZE"])
train_file_chunks = np.array_split(train_files, num_chunks)

print(f"Training files split into {len(train_file_chunks)} chunks of ~{CONFIG['CHUNK_SIZE']} files each")

# Load first chunk to get sample counts and class weights
print("Loading first training chunk for initialization...")
X_chunk, y_chunk = load_chunk_parallel(train_file_chunks[0], saved_feature_cols, CONFIG)
print(f"First chunk: {X_chunk.shape}")

# --- 6. CLASS WEIGHTS (from first chunk - representative sample) ---
pos = np.sum(y_chunk)
neg = len(y_chunk) - pos
total = len(y_chunk)

if pos > 0:
    weight_0 = (1 / neg) * (total / 2.0)
    weight_1 = (1 / pos) * (total / 2.0)
else:
    weight_0, weight_1 = 0.5, 10.0

class_weight = {0: weight_0, 1: weight_1}
print(f"Class Weights -> 0: {weight_0:.2f} | 1: {weight_1:.2f}")

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
    print("Building Distributed JACKPOT Model...")
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

best_val_auc = 0.0
patience_counter = 0
patience = 10

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
        
        # Load chunk
        X_chunk, y_chunk = load_chunk_parallel(file_chunk, saved_feature_cols, CONFIG)
        
        if X_chunk is None:
            continue
        
        # Create dataset for this chunk
        chunk_ds = tf.data.Dataset.from_tensor_slices((X_chunk, y_chunk))
        chunk_ds = chunk_ds.shuffle(buffer_size=min(len(X_chunk), 100_000)) \
                           .batch(CONFIG["BATCH_SIZE"]) \
                           .prefetch(tf.data.AUTOTUNE)
        
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
    
    # Early stopping and checkpointing
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        patience_counter = 0
        model.save(CONFIG["MODEL_PATH"])
        print(f"  ✓ New best model saved! (AUC: {val_auc:.4f})")
    else:
        patience_counter += 1
        print(f"  No improvement (patience: {patience_counter}/{patience})")
    
    if patience_counter >= patience:
        print(f"\n  Early stopping triggered!")
        break

print(f"\n{'='*60}")
print(f"JACKPOT MODEL TRAINING COMPLETE!")
print(f"Best Validation AUC: {best_val_auc:.4f}")
print(f"Model saved: {CONFIG['MODEL_PATH']}")
print(f"{'='*60}")

# --- 10. SAVE SCALER ---
print("\nFitting and saving global scaler...")
sample_files = random.sample(all_files, min(200, len(all_files)))
scaler_data = []
for f in sample_files:
    try:
        df = pd.read_csv(f)
        # Use the same feature columns that were locked during training
        df_feat = df.reindex(columns=saved_feature_cols, fill_value=0)
        df_feat = df_feat.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Ensure all columns are numeric (convert any string columns to numeric or drop them)
        df_feat = df_feat.select_dtypes(include=[np.number])
        
        if not df_feat.empty:
            scaler_data.append(df_feat.values)
    except Exception as e:
        continue

if scaler_data:
    scaler_array = np.concatenate(scaler_data, axis=0)
    global_scaler = StandardScaler()
    global_scaler.fit(scaler_array)
    joblib.dump(global_scaler, CONFIG["SCALER_PATH"])
    print(f"Global scaler saved: {CONFIG['SCALER_PATH']}")
else:
    print("Warning: No data available for scaler fitting.")

# --- 10. SAVE SCALER ---
print("\nFitting and saving global scaler...")
sample_files = random.sample(all_files, min(200, len(all_files)))
scaler_data = []
for f in sample_files:
    try:
        df = pd.read_csv(f)
        # Use the same feature columns that were locked during training
        df_feat = df.reindex(columns=saved_feature_cols, fill_value=0)
        df_feat = df_feat.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Ensure all columns are numeric (convert any string columns to numeric or drop them)
        df_feat = df_feat.select_dtypes(include=[np.number])
        
        if not df_feat.empty:
            scaler_data.append(df_feat.values)
    except Exception as e:
        continue

if scaler_data:
    scaler_array = np.concatenate(scaler_data, axis=0)
    global_scaler = StandardScaler()
    global_scaler.fit(scaler_array)
    joblib.dump(global_scaler, CONFIG["SCALER_PATH"])
    print(f"Global scaler saved: {CONFIG['SCALER_PATH']}")
else:
    print("Warning: No data available for scaler fitting.")
