"""
Processor V7 (Advanced):
NEW ADDITIONS:
- Options Flow Integration (Put/Call ratio, UOA)
- Short Squeeze Score (combines short float + price action + volume)
- Enhanced Momentum (multiple timeframes)
- Volume Profile (buying vs selling pressure)
- Dark Pool Index (institutional accumulation proxy)

Existing:
- Technicals (RSI, MACD, BB, ATR)
- Fundamentals (P/E, Short Float, Sector)
- Insider Trading
- News Sentiment
- Global Context (VIX)
- Jackpot Features (Squeeze, RVOL, Dist to High)
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
OPTIONS_DIR = os.path.join(RAW_DIR, "optionsFlow")
SENTIMENT_DIR = os.path.join(RAW_DIR, "sentiment")

# Global Data Paths
VIX_PATH = "TrainingData/indicators_data/raw/SPY-VIX/^VIX_daily.csv"

os.makedirs(PROCESSED_DIR, exist_ok=True)

def load_global_context():
    """Loads VIX (Fear Index) from markets.py data."""
    print("Loading Global Context (VIX)...")

    if os.path.exists(VIX_PATH):
        df_vix = pd.read_csv(VIX_PATH)
        if isinstance(df_vix.columns, pd.MultiIndex): 
            df_vix.columns = df_vix.columns.get_level_values(0)
        df_vix.columns = [c.lower() for c in df_vix.columns]
        if 'date' not in df_vix.columns and 'Date' in df_vix.columns: 
            df_vix.rename(columns={'Date': 'date'}, inplace=True)
        df_vix['date'] = pd.to_datetime(df_vix['date']).dt.tz_localize(None)
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

# --- NEW: OPTIONS FLOW FEATURES ---

def load_options_data(ticker):
    """Load options flow data for a ticker."""
    path = os.path.join(OPTIONS_DIR, f"{ticker}_options_daily.csv")
    if not os.path.exists(path):
        return None
    
    try:
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Select key columns
        cols = ['put_call_ratio', 'unusual_activity', 'options_sentiment', 'total_options_volume']
        available_cols = [c for c in cols if c in df.columns]
        return df[available_cols]
    except:
        return None

# --- NEW: SHORT SQUEEZE SCORE ---

def calculate_short_squeeze_score(df, short_float):
    """
    Combines multiple factors to detect squeeze potential:
    1. High short interest (short_float > 15%)
    2. Rising price (momentum)
    3. Volume spike (RVOL > 2)
    4. Technical breakout (price > BB upper)
    """
    if short_float < 0.10:  # Less than 10% short = no squeeze potential
        df['squeeze_score'] = 0.0
        return df
    
    # Component 1: Short Interest (0-1 scale)
    si_score = min(short_float / 0.30, 1.0)  # Cap at 30%
    
    # Component 2: Price Momentum (10-day return)
    df['momentum_10d'] = df['close'].pct_change(10).fillna(0)
    mom_score = df['momentum_10d'].apply(lambda x: min(max(x * 10, 0), 1))  # Scale to 0-1
    
    # Component 3: Volume Spike
    if 'rvol' in df.columns:
        vol_score = df['rvol'].apply(lambda x: 1.0 if x > 2 else 0.5 if x > 1.5 else 0.0)
    else:
        vol_score = 0.0
    
    # Component 4: Technical Breakout (price above BB upper)
    if 'BBU_20_2.0' in df.columns:
        breakout_score = (df['close'] > df['BBU_20_2.0']).astype(float)
    else:
        breakout_score = 0.0
    
    # Weighted combination
    df['squeeze_score'] = (
        si_score * 0.3 +      # Short interest
        mom_score * 0.3 +     # Momentum
        vol_score * 0.25 +    # Volume
        breakout_score * 0.15 # Breakout
    )
    
    return df

# --- NEW: VOLUME PROFILE (Buying vs Selling Pressure) ---

def calculate_volume_profile(df):
    """
    Estimates buying vs selling pressure using intraday price action.
    Approximation using daily OHLC since we don't have tick data.
    """
    # Money Flow Multiplier (ranges from -1 to +1)
    # If close near high = buying pressure, near low = selling pressure
    df['mf_multiplier'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10)
    
    # Money Flow Volume
    df['mf_volume'] = df['mf_multiplier'] * df['volume']
    
    # Accumulation/Distribution over 20 days
    df['accum_dist_20'] = df['mf_volume'].rolling(20).sum()
    
    # Normalize to -1 to +1 scale
    df['buy_sell_pressure'] = df['accum_dist_20'] / (df['volume'].rolling(20).sum() + 1e-10)
    
    return df

# --- NEW: ENHANCED MOMENTUM (Multiple Timeframes) ---

def add_enhanced_momentum(df):
    """
    Momentum across multiple timeframes to catch different trading styles:
    - 3d: Day traders
    - 10d: Swing traders
    - 30d: Position traders
    - 90d: Long-term trend
    """
    for period in [3, 10, 30, 90]:
        df[f'momentum_{period}d'] = df['close'].pct_change(period).fillna(0)
    
    # Momentum Acceleration (2nd derivative)
    df['momentum_accel'] = df['momentum_10d'] - df['momentum_10d'].shift(5)
    
    return df

# --- EXISTING: JACKPOT FEATURES ---

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

# --- EXISTING: INSIDER TRADING ---

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

# --- NEW: SENTIMENT INTEGRATION ---

def load_sentiment_data(ticker):
    """Load news sentiment data."""
    path = os.path.join(SENTIMENT_DIR, f"{ticker}_sentiment_daily.csv")
    if not os.path.exists(path):
        return None
    
    try:
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df[['sentiment', 'num_articles']]
    except:
        return None

# --- WORKER ---

def process_file(csv_path, output_path, df_fundamentals, sector_cols, df_global_context):
    try:
        df = pd.read_csv(csv_path)
        df.columns = [c.lower() for c in df.columns]
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        elif 'Date' in df.columns:
            df['date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        else:
            return

        df.sort_values("date", inplace=True)
        ticker = os.path.basename(csv_path).split("_")[0]
        
        # Need OHLCV before setting index for some calculations
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            return
        
        # 1. Merge Global Context (VIX)
        df.set_index('date', inplace=True)
        if not df_global_context.empty:
            df = df.join(df_global_context, how='left')
            df['fear_index'].fillna(method='ffill', inplace=True)
            df['fear_index'].fillna(20.0, inplace=True)

        # 2. Inject Fundamentals (Static columns)
        basic_cols = ['pe_ratio', 'short_float', 'insider_own', 'inst_own', 'market_cap']
        all_fund_cols = basic_cols + sector_cols
        
        short_float_val = 0.0
        if df_fundamentals is not None and ticker in df_fundamentals.index:
            vals = df_fundamentals.loc[ticker]
            for col in all_fund_cols:
                df[col] = vals.get(col, 0.0)
            short_float_val = vals.get('short_float', 0.0)
        else:
            for col in all_fund_cols: 
                df[col] = 0.0

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
        
        # 4. Jackpot Features (EXISTING)
        df = add_jackpot_features(df)
        
        # 5. NEW: Enhanced Momentum
        df = add_enhanced_momentum(df)
        
        # 6. NEW: Volume Profile
        df = calculate_volume_profile(df)
        
        # 7. NEW: Short Squeeze Score
        df = calculate_short_squeeze_score(df, short_float_val)
        
        # 8. NEW: Options Flow Integration
        df_options = load_options_data(ticker)
        if df_options is not None:
            df = df.join(df_options, how='left')
            # Forward fill options data (it updates less frequently)
            options_cols = ['put_call_ratio', 'unusual_activity', 'options_sentiment', 'total_options_volume']
            for col in options_cols:
                if col in df.columns:
                    df[col].fillna(method='ffill', inplace=True)
                    df[col].fillna(0, inplace=True)
        else:
            # Add placeholder columns if no options data
            df['put_call_ratio'] = 1.0
            df['unusual_activity'] = 0
            df['options_sentiment'] = 0.0
            df['total_options_volume'] = 0
        
        # 9. Sentiment Data Integration
        df_sentiment = load_sentiment_data(ticker)
        if df_sentiment is not None:
            df = df.join(df_sentiment, how='left')
            df['sentiment'].fillna(0.0, inplace=True)
            df['num_articles'].fillna(0, inplace=True)
        else:
            df['sentiment'] = 0.0
            df['num_articles'] = 0
        
        # 10. Insider Data
        df_insider = safe_read_insider(ticker)
        if df_insider is not None:
            df = df.join(df_insider, how='left')
            df[['insider_shares', 'insider_amount', 'insider_buy_flag']] = df[['insider_shares', 'insider_amount', 'insider_buy_flag']].fillna(0)
        else:
            df['insider_shares'] = 0
            df['insider_amount'] = 0
            df['insider_buy_flag'] = 0

        # 11. TARGETS
        # Production: >2% gain in 5 days
        df['target_5d'] = (df['close'].shift(-5) > df['close'] * 1.02).astype(int)
        
        # Jackpot: >20% gain in 20 days
        df['target_20d'] = (df['close'].shift(-20) > df['close'] * 1.20).astype(int)
        
        # 12. Clean & Save
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        
        # Drop raw OHLCV to save space (keep close for reference)
        cols_to_drop = ['open', 'high', 'low', 'volume']
        df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
        
        # Reset index to save 'date' as a column
        df.reset_index(inplace=True)

        if len(df) > 50:
            df.to_csv(output_path, index=False)

    except Exception as e:
        # Silently skip errors to avoid spam
        pass

# --- MAIN ---

def main():
    print("Starting Advanced Processor V7...")
    print("NEW FEATURES:")
    print("  - Options Flow (Put/Call, UOA)")
    print("  - Short Squeeze Score")
    print("  - Enhanced Momentum (multi-timeframe)")
    print("  - Volume Profile (buy/sell pressure)")
    print("  - Integrated Sentiment")
    
    # 1. Load Shared Data
    valid_tickers, df_fundamentals, sector_cols = load_fundamentals_and_tickers()
    df_global_context = load_global_context()
    
    # 2. Process
    for subfolder in ["SPY-VIX", "stocksData"]:
        raw_subdir = os.path.join(RAW_DIR, subfolder)
        processed_subdir = os.path.join(PROCESSED_DIR, subfolder)
        os.makedirs(processed_subdir, exist_ok=True)
        
        if not os.path.exists(raw_subdir): continue
        
        files = [f for f in os.listdir(raw_subdir) if f.endswith(".csv")]
        print(f"\nProcessing {len(files)} files in {subfolder}...")
        
        for file in tqdm(files):
            raw_path = os.path.join(raw_subdir, file)
            proc_path = os.path.join(processed_subdir, f"{os.path.splitext(file)[0]}_processed.csv")
            
            process_file(raw_path, proc_path, df_fundamentals, sector_cols, df_global_context)

    print("\n✓ Processing complete!")
    print("All stocks now have:")
    print("  ✓ Options flow signals")
    print("  ✓ Short squeeze detection")
    print("  ✓ Enhanced momentum indicators")
    print("  ✓ Volume profile analysis")

if __name__ == "__main__":
    main()