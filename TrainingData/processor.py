"""
The purpose of this script is to process the raw data found in the indicators_data/raw folder
and place them in the indicators_data/processed folder.
"""

import os
import pandas as pd
import numpy as np
import warnings
from datetime import datetime

# --- CONFIG & SAFETY ---
# Silence annoying Pandas warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

RAW_DIR = "TrainingData/indicators_data/raw"
PROCESSED_DIR = "TrainingData/indicators_data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

def safe_log_return(series_curr, series_prev):
    """Calculates log returns safely, handling zeros/negative values."""
    # Prevent divide by zero or log of zero/negative
    with np.errstate(divide='ignore', invalid='ignore'):
        val = np.log(series_curr / series_prev)
    return val.replace([np.inf, -np.inf], np.nan)

def safe_read_insider(insider_path):
    """Reads and aggregates insider trading data safely."""
    try:
        df_insider = pd.read_csv(insider_path)
        # Handle date formats safely
        df_insider['date'] = pd.to_datetime(
            df_insider['date'].astype(str).str[:10], errors='coerce'
        )
        
        # Drop invalid dates
        df_insider = df_insider.dropna(subset=['date'])
        
        # Group by date
        grouped = df_insider.groupby('date').agg({
            'shares': 'sum',
            'amount': 'sum'
        }).reset_index()
        
        # Logic: 1 (Buy), 0 (Sell), -1 (Neutral/None)
        grouped['insider_buy_flag'] = grouped['shares'].apply(lambda s: 1 if s > 0 else (0 if s < 0 else -1))
        
        grouped = grouped.rename(columns={
            'shares': 'insider_shares',
            'amount': 'insider_amount'
        })
        return grouped[['date', 'insider_shares', 'insider_amount', 'insider_buy_flag']]
        
    except Exception as e:
        # print(f"[WARNING] Failed to process insider file {insider_path}: {e}")
        return pd.DataFrame(columns=['date', 'insider_shares', 'insider_amount', 'insider_buy_flag'])

def process_file(csv_path, output_path):
    try:
        df = pd.read_csv(csv_path)
        
        # Standardize column names
        df.columns = [c.lower() for c in df.columns]
        
        if 'date' not in df.columns:
            # print(f"Skipping {csv_path}: No date column")
            return

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values("date").reset_index(drop=True)

        # --- 1. Log Returns (Safe Mode) ---
        df["YesterdayClose"] = df["close"].shift(1)
        df["YesterdayOpenLogR"]  = safe_log_return(df["open"], df["open"].shift(1))
        df["YesterdayHighLogR"]  = safe_log_return(df["high"], df["high"].shift(1))
        df["YesterdayLowLogR"]   = safe_log_return(df["low"],  df["low"].shift(1))
        df["YesterdayVolumeLogR"] = safe_log_return(df["volume"], df["volume"].shift(1))
        df["YesterdayCloseLogR"] = safe_log_return(df["close"], df["YesterdayClose"])

        # --- 2. Moving Averages ---
        df["MA10"] = df["close"].rolling(window=10).mean()
        df["MA20"] = df["close"].rolling(window=20).mean()
        df["MA30"] = df["close"].rolling(window=30).mean()

        # --- 3. Date Features ---
        df["DayOfWeek"] = df["date"].dt.weekday
        df["DayOfMonth"] = df["date"].dt.day
        df["MonthNumber"] = df["date"].dt.month

        # --- 4. EMAs ---
        df["EMA10"] = df["close"].ewm(span=10, adjust=False).mean()
        df["EMA30"] = df["close"].ewm(span=30, adjust=False).mean()

        # --- 5. RSI ---
        delta = df["close"].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(window=14).mean()
        avg_loss = pd.Series(loss).rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df["RSI"] = 100 - (100 / (1 + rs))

        # --- 6. MACD ---
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = ema12 - ema26
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

        # --- 7. Bollinger Bands ---
        ma20 = df["close"].rolling(window=20).mean()
        std20 = df["close"].rolling(window=20).std()
        df["BollingerUpper"] = ma20 + 2 * std20
        df["BollingerLower"] = ma20 - 2 * std20

        # --- 8. Rolling Volatility (Fixed fill_method) ---
        # Note: fill_method=None prevents the FutureWarning
        pct_change = df["close"].pct_change(fill_method=None)
        
        df["Volatility_10"] = pct_change.rolling(window=10).std()
        df["Volatility_20"] = pct_change.rolling(window=20).std()
        df["Volatility_30"] = pct_change.rolling(window=30).std()
        
        df['volatility_5d'] = pct_change.rolling(5).std() * np.sqrt(252)
        df['volatility_20d'] = pct_change.rolling(20).std() * np.sqrt(252)

        # --- 9. Other Indicators ---
        # OBV
        df["OBV"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()

        # Z-Score
        df["ZScore"] = (df["close"] - ma20) / std20

        # Overnight Gap
        df['overnight_gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        
        # Abnormal Volume
        rolling_vol = df['volume'].rolling(20)
        df['abnormal_vol'] = (df['volume'] - rolling_vol.mean()) / rolling_vol.std()
        
        # Momentum
        df['momentum_5d'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_20d'] = df['close'] / df['close'].shift(20) - 1
        
        # Skew
        df['skew_5d'] = pct_change.rolling(5).skew()
        
        # Intraday Range
        df['intraday_range'] = (df['high'] - df['low']) / df['close']

        # --- 10. Merge Insider Data ---
        ticker = os.path.basename(csv_path).split("_")[0]
        insider_path = os.path.join(RAW_DIR, "insiderBuying", f"{ticker}_insider_trades_daily.csv")
        
        if os.path.exists(insider_path):
            df_insider = safe_read_insider(insider_path)
            df = df.merge(
                df_insider,
                on="date", how="left"
            )
            df["insider_shares"] = df["insider_shares"].fillna(0)
            df["insider_amount"] = df["insider_amount"].fillna(0)
            df["insider_buy_flag"] = df["insider_buy_flag"].fillna(-1).astype(int)
        else:
            df["insider_shares"] = 0
            df["insider_amount"] = 0
            df["insider_buy_flag"] = -1

        # --- 11. Merge Sentiment Data ---
        sentiment_path = os.path.join(RAW_DIR, "sentiment", f"{ticker}_sentiment_daily.csv")

        if os.path.exists(sentiment_path):
            try:
                df_sentiment = pd.read_csv(sentiment_path)
                df_sentiment['date'] = pd.to_datetime(df_sentiment['date'])
                
                df = df.merge(
                    df_sentiment[["date", "sentiment", "num_articles"]],
                    on="date", how="left"
                )
                df["sentiment"] = df["sentiment"].fillna(0)
                df["num_articles"] = df["num_articles"].fillna(0)
            except Exception:
                df["sentiment"] = 0
                df["num_articles"] = 0
        else:
            df["sentiment"] = 0
            df["num_articles"] = 0

        # Sentiment Change
        df['sentiment_change'] = df['sentiment'] - df['sentiment'].shift(1)

        # --- 12. Final Cleanup (Crucial for AI) ---
        # Replace Infinity with NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Drop rows with NaN (due to rolling windows/log returns)
        df.dropna(inplace=True)
        
        # Target for Prediction (Forward looking 5 days)
        # 1 if price goes up > 2%, else 0
        df['target_5d'] = (df['close'].shift(-5) > df['close'] * 1.02).astype(int)

        # Drop raw price columns if you want only indicators
        # Note: Keeping 'close' is usually good for reference, but respecting your drop list
        cols_to_drop = ['open', 'high', 'low', 'volume']
        df = df.drop([c for c in cols_to_drop if c in df.columns], axis=1)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        # print(f"Processed: {output_path}")

    except Exception as e:
        print(f"Error processing {csv_path}: {e}")

def check_missing_today():
    today = pd.Timestamp(datetime.today().date())
    print("\n[INFO] Checking which files are missing today's data...\n")
    missing = []

    for subfolder in ["SPY-VIX", "stocksData"]:
        processed_subdir = os.path.join(PROCESSED_DIR, subfolder)
        if not os.path.exists(processed_subdir):
            continue
        for file in os.listdir(processed_subdir):
            if not file.endswith("_processed.csv"):
                continue
            file_path = os.path.join(processed_subdir, file)
            try:
                df = pd.read_csv(file_path, parse_dates=["date"])
                if df.empty:
                    missing.append((file, "EMPTY"))
                    continue
                last_date = df["date"].max()
                if last_date.date() < today.date():
                    missing.append((file, last_date.date()))
            except Exception:
                pass

    if missing:
        print(f"{len(missing)} files are outdated (do not contain today's data).")
        # Uncomment to list them all
        # for filename, last_date in missing[:5]:
        #     print(f" - {filename}: Last date = {last_date}")
        # if len(missing) > 5: print(" ... and others.")
    else:
        print("All files contain today's data.")

def main():
    print("Starting Data Processor...")
    for subfolder in ["SPY-VIX", "stocksData"]:
        raw_subdir = os.path.join(RAW_DIR, subfolder)
        processed_subdir = os.path.join(PROCESSED_DIR, subfolder)
        os.makedirs(processed_subdir, exist_ok=True)

        if not os.path.exists(raw_subdir):
            print(f"Directory not found: {raw_subdir}")
            continue

        files = [f for f in os.listdir(raw_subdir) if f.endswith(".csv")]
        print(f"Processing {len(files)} files in {subfolder}...")

        for i, file in enumerate(files):
            raw_file_path = os.path.join(raw_subdir, file)
            processed_file_path = os.path.join(processed_subdir, f"{os.path.splitext(file)[0]}_processed.csv")
            process_file(raw_file_path, processed_file_path)
            
            if i % 100 == 0:
                print(f"   Processed {i}/{len(files)}...")
    
    print("Processing Complete.")

if __name__ == "__main__":
    main()
    check_missing_today()