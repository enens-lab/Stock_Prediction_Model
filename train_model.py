import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv1D, LSTM, Dense, Dropout, BatchNormalization,
    Input, MultiHeadAttention, LayerNormalization, Add
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import mixed_precision
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
    "MODEL_PATH": Path("TrainingData/models/lstm_production.keras"),
    "FEATURE_MAP_PATH": Path("TrainingData/models/feature_columns_production.json"),
    "SCALER_PATH": Path("TrainingData/models/scaler_production.joblib"),
    "SEQ_LEN": 60,
    "PRED_HORIZON": 5,
    "BATCH_SIZE": GLOBAL_BATCH_SIZE,
    "EPOCHS": 60,
}
os.makedirs(CONFIG["MODEL_DIR"], exist_ok=True)

#mixed_precision.set_global_policy('mixed_float16')

print(f"Starting PRODUCTION Training (Global Batch Size: {CONFIG['BATCH_SIZE']})...")

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
cols_to_drop = ['date', 'target_5d', 'open', 'high', 'low', 'close', 'volume', 'ticker', 'target']
saved_feature_cols = [c for c in first_df.columns if c not in cols_to_drop]

print(f"LOCKED Features ({len(saved_feature_cols)}):")
print(f"   {saved_feature_cols[:5]} ...")

with open(CONFIG["FEATURE_MAP_PATH"], "w") as f_json:
    json.dump(saved_feature_cols, f_json)

# --- 3. PARALLEL DATA LOADER (ThreadPoolExecutor) ---
def process_single_file(f, feature_cols, config):
    try:
        df = pd.read_csv(f)
        if len(df) < config["SEQ_LEN"] + config["PRED_HORIZON"]:
            return None, None

        df_features = df.reindex(columns=feature_cols, fill_value=0)
        df_features = df_features.replace([np.inf, -np.inf], np.nan).fillna(0)

        scaler = StandardScaler()
        X_raw = scaler.fit_transform(df_features.values).astype(np.float16)

        if 'target_5d' in df.columns:
            targets = df['target_5d'].fillna(0).astype(np.int8).values
        else:
            close = df['close'].values
            future_close = np.roll(close, -config["PRED_HORIZON"])
            returns = (future_close - close) / close
            returns[-config["PRED_HORIZON"]:] = 0
            targets = (returns > 0.02).astype(np.int8)

        X_local = []
        y_local = []
        limit = len(X_raw) - config["PRED_HORIZON"]

        for i in range(config["SEQ_LEN"], limit):
            X_local.append(X_raw[i-config["SEQ_LEN"]:i])
            y_local.append(targets[i])

        return X_local, y_local
    except Exception:
        return None, None

def load_files_parallel(files, feature_cols, config, desc="Loading"):
    print(f"{desc} ({len(files)} files)...")
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_single_file, f, feature_cols, config) for f in files]
        results = [fut.result() for fut in futures]

    X_all, y_all = [], []
    for X_part, y_part in results:
        if X_part is not None and len(X_part) > 0:
            X_all.extend(X_part)
            y_all.extend(y_part)

    if not X_all:
        raise ValueError(f"No valid data extracted from {desc} files.")

    X = np.array(X_all, dtype=np.float16)
    y = np.array(y_all, dtype=np.int8)
    del X_all, y_all, results
    gc.collect()
    return X, y

# --- LOAD DATA ---
X_train, y_train = load_files_parallel(train_files, saved_feature_cols, CONFIG, "Training Data")
print(f"Training Data Ready: {X_train.shape}")

X_val, y_val = load_files_parallel(val_files, saved_feature_cols, CONFIG, "Validation Data")
print(f"Validation Data Ready: {X_val.shape}")

# --- 4. TF DATASET PIPELINE ---
shuffle_buffer = min(len(X_train), 2_000_000)

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.cache() \
                   .shuffle(buffer_size=shuffle_buffer) \
                   .batch(CONFIG["BATCH_SIZE"]) \
                   .prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_ds = val_ds.cache() \
               .batch(CONFIG["BATCH_SIZE"]) \
               .prefetch(tf.data.AUTOTUNE)

# Auto-sharding for multi-GPU
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
train_ds = train_ds.with_options(options)
val_ds = val_ds.with_options(options)

# --- 5. CLASS WEIGHTS ---
pos = np.sum(y_train)
neg = len(y_train) - pos
total = len(y_train)

if pos > 0:
    weight_0 = (1 / neg) * (total / 2.0)
    weight_1 = (1 / pos) * (total / 2.0)
else:
    weight_0, weight_1 = 0.5, 10.0

class_weight = {0: weight_0, 1: weight_1}
print(f"Class Weights -> 0: {weight_0:.2f} | 1: {weight_1:.2f}")

# --- 6. LEARNING RATE SCHEDULE (Cosine Decay with Warmup) ---
steps_per_epoch = max(1, len(X_train) // CONFIG["BATCH_SIZE"])

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

warmup_steps = steps_per_epoch  # 1 epoch warmup
cosine_decay_steps = steps_per_epoch * 5

lr_schedule = WarmupCosineDecay(
    warmup_steps=warmup_steps,
    total_decay_steps=cosine_decay_steps,
    initial_lr=5e-4,        # Changed from 1e-3
    warmup_start_lr=1e-6,   # Changed from 1e-5
)

# --- 7. MODEL ARCHITECTURE (Functional API for Attention) ---
n_features = X_train.shape[2]

with strategy.scope():
    print("Building Distributed Model...")
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

    # Around line 197, replace the optimizer line:
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)  # Add gradient clipping
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy', 'AUC'])

model.summary()

# --- 8. TRAINING ---
print("Training PRODUCTION MODEL...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=CONFIG["EPOCHS"],
    callbacks=[
        EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss'),
        ModelCheckpoint(CONFIG["MODEL_PATH"], save_best_only=True, monitor='val_loss')
    ],
    class_weight=class_weight
)

print(f"PRODUCTION MODEL SAVED: {CONFIG['MODEL_PATH']}")

# --- 9. SAVE SCALER (for Pythia export) ---
print("Fitting and saving global scaler...")
sample_files = random.sample(all_files, min(200, len(all_files)))
scaler_data = []
for f in sample_files:
    try:
        df = pd.read_csv(f)
        df_feat = df.reindex(columns=saved_feature_cols, fill_value=0)
        df_feat = df_feat.replace([np.inf, -np.inf], np.nan).fillna(0)
        scaler_data.append(df_feat.values)
    except Exception:
        continue

scaler_array = np.concatenate(scaler_data, axis=0)
global_scaler = StandardScaler()
global_scaler.fit(scaler_array)
joblib.dump(global_scaler, CONFIG["SCALER_PATH"])
print(f"Global scaler saved: {CONFIG['SCALER_PATH']}")
