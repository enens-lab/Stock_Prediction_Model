import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import glob
import os
from datetime import datetime, timedelta

# --- CONFIG ---
CONFIG = {
    "DATA_DIR": Path("TrainingData/indicators_data/processed/stocksData"),
    "MODEL_PATH": Path("lstm_model.h5"),
    "SEQ_LEN": 60,       
    "MIN_CONFIDENCE": 0.80 
}

def main():
    print("Starting Daily Scanner...")
    
    if not os.path.exists(CONFIG["MODEL_PATH"]):
        print("Model not found! Train it first.")
        return
    
    print(f"Loading Model: {CONFIG['MODEL_PATH']}...")
    model = load_model(CONFIG["MODEL_PATH"])
    
    all_files = glob.glob(str(CONFIG["DATA_DIR"] / "*.csv"))
    print(f"Scanning {len(all_files)} stocks...")
    
    recommendations = []
    
    # Calculate Cutoff (Ignore data older than 4 days)
    cutoff_date = datetime.now() - timedelta(days=4)
    print(f"Ignoring data older than: {cutoff_date.strftime('%Y-%m-%d')}")
    
    for idx, f in enumerate(all_files):
        try:
            df = pd.read_csv(f)
            if len(df) < CONFIG["SEQ_LEN"]: continue
            
            # --- STALE DATA GUARD ---
            last_date_str = df['date'].iloc[-1]
            last_date_dt = pd.to_datetime(last_date_str)
            
            if last_date_dt < cutoff_date:
                continue
            # ------------------------

            cols_to_drop = ['date', 'target_5d', 'open', 'high', 'low', 'close', 'volume']
            feature_cols = [c for c in df.columns if c not in cols_to_drop]
            
            # Clean
            df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
            df.dropna(subset=feature_cols, inplace=True)
            
            scaler = StandardScaler()
            X_raw = scaler.fit_transform(df[feature_cols].values)
            
            last_seq = X_raw[-CONFIG["SEQ_LEN"]:]
            last_seq = np.expand_dims(last_seq, axis=0)
            
            prob = model.predict(last_seq, verbose=0)[0][0]
            
            if prob >= CONFIG["MIN_CONFIDENCE"]:
                ticker = Path(f).stem.replace('_daily_processed', '')
                last_close = df['close'].iloc[-1]
                
                recommendations.append({
                    "Date": last_date_str,
                    "Ticker": ticker,
                    "Confidence": prob,
                    "Price": last_close
                })
                
        except Exception:
            continue
            
        if idx % 500 == 0:
            print(f"   Scanned {idx}/{len(all_files)}...")

    if not recommendations:
        print("No trades found (or all data was stale).")
    else:
        rec_df = pd.DataFrame(recommendations)
        rec_df = rec_df.sort_values(by="Confidence", ascending=False).head(20)
        
        print("\n" + "="*50)
        print(f"TOP AI PICKS")
        print("="*50)
        print(rec_df.to_string(index=False))
        print("="*50)
        
        rec_df.to_csv("daily_picks.csv", index=False)
        print("Saved to daily_picks.csv")

if __name__ == "__main__":
    main()
