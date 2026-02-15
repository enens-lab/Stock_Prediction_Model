import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import os
import glob
import json
import argparse
import datetime
from tqdm import tqdm

# --- CONFIGURATION HUB ---
MODES = {
    "consistency": {
        "model_file": "lstm_production.keras",
        "map_file": "feature_columns_production.json",
        "horizon": 5,
        "output_file": "daily_picks_consistency.csv",
        "min_prob": 0.60,
        "desc": "Consistency (Aim: >2% in 5 Days)"
    },
    "jackpot": {
        "model_file": "lstm_jackpot.keras",
        "map_file": "feature_columns_jackpot.json",
        "horizon": 20,
        "output_file": "daily_picks_jackpot.csv",
        "min_prob": 0.55,
        "desc": "Jackpot Hunter (Aim: >20% in 20 Days)"
    }
}

DATA_DIR = "TrainingData/indicators_data/processed/stocksData"
MODEL_DIR = "TrainingData/models"
REPORTS_DIR = "TrainingData/reports"
SEQ_LEN = 60
BATCH_SIZE = 1024  # Process 1024 stocks at a time

def load_data_and_predict(mode_config):
    model_path = os.path.join(MODEL_DIR, mode_config["model_file"])
    map_path = os.path.join(MODEL_DIR, mode_config["map_file"])
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    print(f"Loading Model: {mode_config['desc']}...")
    try:
        model = load_model(model_path, compile=False) 
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    with open(map_path, "r") as f:
        feature_cols = json.load(f)
    print(f"Features Locked: {len(feature_cols)}")

    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    print(f"Scanning {len(files)} stocks...")
    
    results = []
    
    # --- BATCH PROCESSING LOOP ---
    # Instead of predict() per file, we bundle them to maximize GPU speed
    
    batch_X = []
    batch_meta = [] # Stores ticker, price, etc. to map back results
    
    for i, file in enumerate(tqdm(files, desc="⚡ Processing Stocks")):
        try:
            df = pd.read_csv(file)
            ticker = os.path.basename(file).split('_')[0]
            
            if len(df) < SEQ_LEN: continue
            
            # 1. Feature Alignment
            df_features = df.reindex(columns=feature_cols, fill_value=0)
            df_features = df_features.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # 2. Scale (Local Fit)
            # We fit on the last year (252 days) to normalize this specific stock's range
            fit_window = df_features.iloc[-252:].values if len(df) > 252 else df_features.values
            scaler = StandardScaler()
            scaler.fit(fit_window)
            
            # 3. Prepare Input Sequence
            recent_data = df_features.iloc[-SEQ_LEN:].values
            X_input = scaler.transform(recent_data) # Shape: (60, features)
            
            # 4. Add to Batch
            batch_X.append(X_input)
            
            # Store metadata for reporting later
            last_price = df['close'].iloc[-1]
            rvol = df['rvol'].iloc[-1] if 'rvol' in df.columns else 0
            squeeze = df['bb_width'].iloc[-1] if 'bb_width' in df.columns else 0
            fear = df['fear_index'].iloc[-1] if 'fear_index' in df.columns else 0

            batch_meta.append({
                "Ticker": ticker,
                "Price": last_price,
                "RVOL": rvol,
                "Squeeze": squeeze,
                "VIX": fear
            })
            
            # 5. EXECUTE BATCH IF FULL
            if len(batch_X) >= BATCH_SIZE or i == len(files) - 1:
                if not batch_X: continue
                
                # Convert list to array: (Batch_Size, 60, Features)
                X_batch_np = np.array(batch_X, dtype=np.float32)
                
                # GPU INFERENCE BLAST 
                preds = model.predict(X_batch_np, verbose=0)
                
                # Map predictions back to metadata
                for j, prob in enumerate(preds):
                    probability = prob[0]
                    if probability > mode_config["min_prob"]:
                        meta = batch_meta[j]
                        meta["Probability"] = round(probability * 100, 2)
                        meta["RVOL"] = round(meta["RVOL"], 2)
                        meta["Squeeze"] = round(meta["Squeeze"], 4)
                        meta["VIX"] = round(meta["VIX"], 2)
                        meta["Date"] = datetime.date.today()
                        results.append(meta)
                
                # Clear Batch
                batch_X = []
                batch_meta = []
                
        except Exception:
            continue

    # Sort and Save
    if results:
        df_res = pd.DataFrame(results)
        # Reorder columns for neatness
        cols = ["Ticker", "Date", "Probability", "Price", "RVOL", "Squeeze", "VIX"]
        df_res = df_res[cols].sort_values(by="Probability", ascending=False)
        
        os.makedirs(REPORTS_DIR, exist_ok=True)
        save_path = os.path.join(REPORTS_DIR, mode_config["output_file"])
        df_res.to_csv(save_path, index=False)
        
        print(f"\nTOP 5 PICKS ({mode_config['desc']}):")
        print(df_res.head(5))
        print(f"Full report saved to: {save_path}")
    else:
        print("\nNo stocks met the probability threshold today.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["consistency", "jackpot"], default="consistency", 
                        help="Choose inference mode")
    args = parser.parse_args()
    
    config = MODES[args.mode]
    load_data_and_predict(config)