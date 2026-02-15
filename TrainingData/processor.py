"""
Processor V6 (Production):
- Technicals (RSI, MACD, BB, ATR)
- Fundamentals (P/E, Short Float, Sector One-Hot)
- Insider Trading (Shares, Amount)
- News Sentiment (Daily Sentiment Score)
- GLOBAL CONTEXT: Merges VIX (Fear Index) from markets.py
- JACKPOT FEATURES: Squeeze, RVOL, Dist to High
"""

import os
import pandas as pd
import numpy as np
#import pandas_ta as ta
import warnings
from tqdm import tqdm

# --- CONFIG ---
warnings.filterwarnings('ignore')

RAW_DIR = "TrainingData/indicators_data/raw"
PROCESSED_DIR = "TrainingData/indicators_data/processed"
FUNDAMENTALS_PATH = "TrainingData/indicators_data/raw/finviz_fundamentals.csv"
STOCKS_DATA_DIR = os.path.join(RAW_DIR, "stocksData")

# Global Data Paths
VIX_PATH = "TrainingData/indicators_data/raw/SPY-VIX/^VIX_daily.csv"

os.makedirs(PROCESSED_DIR, exist_ok=True)

def load_global_context():
    """
    Loads VIX (Fear Index) from markets.py data.
    Macro/sector sentiment (Google Trends) removed — unreliable signal.
    """
    print("Loading Global Context (VIX)...")

    # Load VIX
    if os.path.exists(VIX_PATH):
        df_vix = pd.read_csv(VIX_PATH)
        if isinstance(df_vix.columns, pd.MultiIndex): df_vix.columns = df_vix.columns.get_level_values(0)
        df_vix.columns = [c.lower() for c in df_vix.columns]
        if 'date' not in df_vix.columns and 'Date' in df_vix.columns: df_vix.rename(columns={'Date': 'date'}, inplace=True)
        df_vix['date'] = pd.to_datetime(df_vix['date'], utc=True).dt.tz_localize(None)
        df_vix.set_index('date', inplace=True)
        df_vix = df_vix[['close']].rename(columns={'close': 'fear_index'})
    else:
        df_vix = pd.DataFrame()

    global_df = df_vix.copy()
    global_df.ffill(inplace=True)
    global_df.fillna(0.0, inplace=True)

    return global_df

def load_fundamentals_and_tickers():
    """Scans for valid tickers and loads Finviz data."""
    if not os.path.exists(STOCKS_DATA_DIR):
        return set(), None, []
    
    files = [f for f in os.listdir(STOCKS_DATA_DIR) if f.endswith("_daily.csv")]
    valid_tickers = set(f.split('_')[0] for f in files)
    print(f"Found {len(valid_tickers)} valid stock files.")
    
    # Load Finviz
    if not os.path.exists(FUNDAMENTALS_PATH):
        return valid_tickers, None, []
    
    try:
        df = pd.read_csv(FUNDAMENTALS_PATH)
        df = df[df['Ticker'].isin(valid_tickers)].copy()
        
        # Map Headers
        rename_map = {
            'Short Float': 'short_float', 'Insider Ownership': 'insider_own', 
            'Institutional Ownership': 'inst_own', 'P/E': 'pe_ratio', 
            'Market Cap': 'market_cap', 'Sector': 'sector'
        }
        df.rename(columns=rename_map, inplace=True)
        
        # Clean Percentages
        for col in ['short_float', 'insider_own', 'inst_own']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip('%').apply(pd.to_numeric, errors='coerce') / 100
        
        if 'pe_ratio' in df.columns:
            df['pe_ratio'] = pd.to_numeric(df['pe_ratio'], errors='coerce')

        # One-Hot Encode Sector
        sector_cols = []
        if 'sector' in df.columns:
            df['sector'].fillna('Unknown', inplace=True)
            dummies = pd.get_dummies(df['sector'], prefix='sec')
            dummies.columns = [c.lower().replace(' ', '_') for c in dummies.columns]
            sector_cols = list(dummies.columns)
            df = pd.concat([df, dummies], axis=1)
        
        df.set_index('Ticker', inplace=True)
        return valid_tickers, df, sector_cols
        
    except Exception as e:
        print(f"Error loading fundamentals: {e}")
        return valid_tickers, None, []

# --- FEATURES ---

def add_jackpot_features(df):
    """Adds Squeeze, RVOL, Dist-to-High"""
    sma = df['close'].rolling(window=20).mean()
    std = df['close'].rolling(window=20).std()
    
    # BB Squeeze
    df['bb_width'] = (((sma + std*2) - (sma - std*2)) / sma).replace([np.inf, -np.inf], 0)
    
    # RVOL (Fuel)
    df['vol_sma_20'] = df['volume'].rolling(window=20).mean()
    df['rvol'] = (df['volume'] / df['vol_sma_20']).fillna(1.0)
    
    # Dist to High
    df['high_52'] = df['close'].rolling(window=252).max()
    df['dist_to_high'] = ((df['close'] - df['high_52']) / df['high_52']).fillna(-1.0)
    
    # Velocity
    df['roc_10'] = df['close'].pct_change(periods=10).fillna(0)
    return df

def safe_read_insider(ticker):
    path = os.path.join(RAW_DIR, "insiderBuying", f"{ticker}_insider_trades_daily.csv")
    if not os.path.exists(path): return None
    try:
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['date'].astype(str).str[:10], errors='coerce')
        df = df.dropna(subset=['date'])
        grouped = df.groupby('date').agg({'shares': 'sum', 'amount': 'sum'}).reset_index()
        grouped['insider_buy_flag'] = grouped['shares'].apply(lambda s: 1 if s > 0 else (0 if s < 0 else -1))
        grouped.rename(columns={'shares': 'insider_shares', 'amount': 'insider_amount'}, inplace=True)
        grouped.set_index('date', inplace=True)
        return grouped
    except:
        return None

# --- WORKER ---

def process_file(csv_path, output_path, df_fundamentals, sector_cols, df_global_context):
    try:
        df = pd.read_csv(csv_path)
        df.columns = [c.lower() for c in df.columns]
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
        elif 'Date' in df.columns:
            df['date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)
        else:
            return

        df.sort_values("date", inplace=True)
        df.set_index('date', inplace=True) # Index by date for easy merging
        ticker = os.path.basename(csv_path).split("_")[0]

        # 1. Merge Global Context (VIX only)
        if not df_global_context.empty:
            df = df.join(df_global_context, how='left')
            df['fear_index'].fillna(method='ffill', inplace=True)
            df['fear_index'].fillna(20.0, inplace=True)

        # 2. Inject Fundamentals (Static columns)
        basic_cols = ['pe_ratio', 'short_float', 'insider_own', 'inst_own', 'market_cap']
        all_fund_cols = basic_cols + sector_cols
        
        if df_fundamentals is not None and ticker in df_fundamentals.index:
            vals = df_fundamentals.loc[ticker]
            for col in all_fund_cols:
                df[col] = vals.get(col, 0.0)
        else:
            for col in all_fund_cols: df[col] = 0.0

        # 3. Technicals
        df["rsi"] = ta.rsi(df["close"], length=14)
        macd = ta.macd(df["close"])
        if macd is not None: df = pd.concat([df, macd], axis=1)
        bb = ta.bbands(df["close"], length=20)
        if bb is not None: df = pd.concat([df, bb], axis=1)
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
        
        # Returns
        df["YesterdayCloseLogR"] = np.log(df["close"] / df["close"].shift(1))
        df["YesterdayVolumeLogR"] = np.log(df["volume"] / df["volume"].shift(1))
        
        # 4. Jackpot Features
        df = add_jackpot_features(df)
        
        # 5. Insider Data
        df_insider = safe_read_insider(ticker)
        if df_insider is not None:
            df = df.join(df_insider, how='left')
            df[['insider_shares', 'insider_amount', 'insider_buy_flag']] = df[['insider_shares', 'insider_amount', 'insider_buy_flag']].fillna(0)
        else:
            df['insider_shares'] = 0
            df['insider_amount'] = 0
            df['insider_buy_flag'] = 0

        # 6. Clean & Save
        # Target: >2% gain in 5 days
        df['target_5d'] = (df['close'].shift(-5) > df['close'] * 1.02).astype(int)
        
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        
        # Drop raw price columns to save space
        cols_to_drop = ['open', 'high', 'low', 'volume']
        df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
        
        # Reset index to save 'date' as a column
        df.reset_index(inplace=True)

        if len(df) > 50:
            df.to_csv(output_path, index=False)

    except Exception as e:
        # print(f"Error {csv_path}: {e}")
        pass

# --- MAIN ---

def main():
    print("Starting Optimized Processor V6...")
    
    # 1. Load Shared Data (The Optimization)
    valid_tickers, df_fundamentals, sector_cols = load_fundamentals_and_tickers()
    df_global_context = load_global_context()
    
    # 2. Process
    for subfolder in ["SPY-VIX", "stocksData"]:
        raw_subdir = os.path.join(RAW_DIR, subfolder)
        processed_subdir = os.path.join(PROCESSED_DIR, subfolder)
        os.makedirs(processed_subdir, exist_ok=True)
        
        if not os.path.exists(raw_subdir): continue
        
        files = [f for f in os.listdir(raw_subdir) if f.endswith(".csv")]
        print(f"Processing {len(files)} files in {subfolder}...")
        
        for file in tqdm(files):
            raw_path = os.path.join(raw_subdir, file)
            proc_path = os.path.join(processed_subdir, f"{os.path.splitext(file)[0]}_processed.csv")
            
            # Pass the loaded dataframes directly!
            process_file(raw_path, proc_path, df_fundamentals, sector_cols, df_global_context)

    print("Done. All stocks now have Fear Index (VIX).")

if __name__ == "__main__":
    main()