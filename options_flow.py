"""
Historical Options Flow Scraper - Builds Complete Time Series

IMPORTANT: This version builds a FULL HISTORICAL dataset, not just today.

What it does:
1. Gets current options chain
2. Calculates daily metrics (Put/Call ratio, volume, OI)
3. Builds time series going back as far as data allows
4. Merges with existing data if available

Limitations:
- Yahoo Finance only provides CURRENT options chain (not historical chains)
- We calculate metrics from TODAY's chain as a proxy for historical sentiment
- For true historical options data, would need paid service ($$$)

Workaround:
- Store TODAY's metrics daily
- Build history over time by running this daily
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from tqdm import tqdm
import time

# --- CONFIG ---
RAW_STOCKS_DIR = 'TrainingData/indicators_data/raw/stocksData'
OPTIONS_DIR = 'TrainingData/indicators_data/raw/optionsFlow'
os.makedirs(OPTIONS_DIR, exist_ok=True)

def get_options_metrics_today(ticker):
    """
    Extracts TODAY's options metrics for a given ticker.
    
    Returns dict with today's metrics or None if no options data.
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Get all expiration dates
        expirations = stock.options
        if not expirations:
            return None
        
        # Focus on near-term options (7-60 days out) - most predictive
        today = datetime.now()
        near_expirations = []
        for exp in expirations:
            exp_date = datetime.strptime(exp, '%Y-%m-%d')
            days_out = (exp_date - today).days
            if 7 <= days_out <= 60:
                near_expirations.append(exp)
        
        if not near_expirations:
            return None
        
        # Aggregate metrics across near-term expirations
        total_call_volume = 0
        total_put_volume = 0
        total_call_oi = 0
        total_put_oi = 0
        
        for exp in near_expirations[:3]:  # Top 3 expirations
            try:
                opt_chain = stock.option_chain(exp)
                
                # Calls
                calls = opt_chain.calls
                if not calls.empty:
                    total_call_volume += calls['volume'].fillna(0).sum()
                    total_call_oi += calls['openInterest'].fillna(0).sum()
                
                # Puts
                puts = opt_chain.puts
                if not puts.empty:
                    total_put_volume += puts['volume'].fillna(0).sum()
                    total_put_oi += puts['openInterest'].fillna(0).sum()
                    
            except:
                continue
        
        # Calculate key metrics
        if total_call_volume + total_put_volume == 0:
            return None
        
        metrics = {
            'date': today.strftime('%Y-%m-%d'),
            'put_call_ratio': total_put_volume / max(total_call_volume, 1),
            'call_volume': int(total_call_volume),
            'put_volume': int(total_put_volume),
            'call_oi': int(total_call_oi),
            'put_oi': int(total_put_oi),
            'total_options_volume': int(total_call_volume + total_put_volume),
        }
        
        return metrics
        
    except Exception as e:
        return None

def backfill_historical_estimates(df, ticker):
    """
    Since Yahoo doesn't provide historical options chains, we:
    1. Use TODAY's put/call ratio as a baseline
    2. Add noise/variation to simulate historical changes
    3. This is an APPROXIMATION - not real historical data
    
    For production use, would need paid data source.
    """
    if df is None or len(df) == 0:
        return df
    
    # Get today's actual value
    today_pcr = df['put_call_ratio'].iloc[-1]
    
    # Generate synthetic historical values with reasonable variation
    # Put/Call ratio typically ranges 0.5-2.0
    np.random.seed(hash(ticker) % 2**32)  # Deterministic per ticker
    
    for i in range(len(df) - 1):
        # Add random walk around today's value
        days_ago = len(df) - 1 - i
        noise_factor = np.random.normal(0, 0.1)  # 10% std dev
        historical_pcr = today_pcr * (1 + noise_factor)
        historical_pcr = np.clip(historical_pcr, 0.3, 3.0)  # Reasonable bounds
        
        df.at[df.index[i], 'put_call_ratio'] = historical_pcr
        
        # Scale volume proportionally
        volume_factor = np.random.uniform(0.5, 1.5)
        df.at[df.index[i], 'total_options_volume'] = int(df['total_options_volume'].iloc[-1] * volume_factor)
    
    return df

def calculate_derived_metrics(df, ticker):
    """
    Calculates derived metrics from options data.
    """
    if df is None or len(df) < 20:
        return df
    
    # Calculate rolling averages (20 days)
    df['avg_volume_20d'] = df['total_options_volume'].rolling(20, min_periods=1).mean()
    
    # Unusual Options Activity (UOA): Volume > 2x average
    df['unusual_activity'] = (df['total_options_volume'] > df['avg_volume_20d'] * 2).astype(int)
    
    # Options Sentiment Score
    # Put/Call < 0.7 = Bullish (more calls = bullish)
    # Put/Call > 1.5 = Bearish (more puts = bearish)
    df['options_sentiment'] = 0.0
    df.loc[df['put_call_ratio'] < 0.7, 'options_sentiment'] = 1.0   # Bullish
    df.loc[df['put_call_ratio'] > 1.5, 'options_sentiment'] = -1.0  # Bearish
    
    return df

def create_full_history_for_ticker(ticker):
    """
    Creates a complete historical options dataset for a ticker.
    
    Strategy:
    1. Load stock price history (we know these dates)
    2. Get TODAY's options metrics
    3. Backfill historical estimates (approximation)
    4. Save complete time series
    """
    save_path = os.path.join(OPTIONS_DIR, f"{ticker}_options_daily.csv")
    
    # Load stock price history to get date range
    stock_file = os.path.join(RAW_STOCKS_DIR, f"{ticker}_daily.csv")
    if not os.path.exists(stock_file):
        return False
    
    try:
        stock_df = pd.read_csv(stock_file)
        
        # Handle multi-level columns from Yahoo Finance
        if isinstance(stock_df.columns, pd.MultiIndex):
            stock_df.columns = stock_df.columns.get_level_values(0)
        
        stock_df.columns = [c.lower() for c in stock_df.columns]
        
        # Get date column
        if 'date' in stock_df.columns:
            stock_df['date'] = pd.to_datetime(stock_df['date'])
        elif 'Date' in stock_df.columns:
            stock_df.rename(columns={'Date': 'date'}, inplace=True)
            stock_df['date'] = pd.to_datetime(stock_df['date'])
        else:
            return False
        
        # Get TODAY's options metrics
        today_metrics = get_options_metrics_today(ticker)
        
        if today_metrics is None:
            # No options data available for this ticker
            return False
        
        # Create dataframe with all historical dates
        date_range = stock_df['date'].values
        
        # Initialize with today's values
        options_df = pd.DataFrame({
            'date': date_range,
            'put_call_ratio': today_metrics['put_call_ratio'],
            'call_volume': today_metrics['call_volume'],
            'put_volume': today_metrics['put_volume'],
            'call_oi': today_metrics['call_oi'],
            'put_oi': today_metrics['put_oi'],
            'total_options_volume': today_metrics['total_options_volume'],
        })
        
        options_df['date'] = pd.to_datetime(options_df['date'])
        
        # Backfill with synthetic variation (approximation)
        options_df = backfill_historical_estimates(options_df, ticker)
        
        # Calculate derived metrics
        options_df = calculate_derived_metrics(options_df, ticker)
        
        # Save
        options_df.to_csv(save_path, index=False)
        return True
        
    except Exception as e:
        return False

def update_existing_ticker(ticker):
    """
    Updates existing options file with today's data.
    (For daily updates after initial backfill)
    """
    save_path = os.path.join(OPTIONS_DIR, f"{ticker}_options_daily.csv")
    
    # Get today's metrics
    metrics = get_options_metrics_today(ticker)
    if metrics is None:
        return False
    
    # Load existing data
    if os.path.exists(save_path):
        try:
            df = pd.read_csv(save_path)
            df['date'] = pd.to_datetime(df['date'])
            
            # Check if today already exists
            today = metrics['date']
            if today in df['date'].values:
                # Update today's row
                df.loc[df['date'] == today, 'put_call_ratio'] = metrics['put_call_ratio']
                df.loc[df['date'] == today, 'total_options_volume'] = metrics['total_options_volume']
            else:
                # Append new row
                df = pd.concat([df, pd.DataFrame([metrics])], ignore_index=True)
            
            # Recalculate derived metrics
            df = calculate_derived_metrics(df, ticker)
            
            # Save
            df.to_csv(save_path, index=False)
            return True
            
        except:
            return False
    else:
        # No existing file, create full history
        return create_full_history_for_ticker(ticker)

def main():
    """
    Main execution: Creates complete historical options dataset.
    """
    print("="*60)
    print("HISTORICAL OPTIONS FLOW BUILDER")
    print("="*60)
    print("\n⚠️  IMPORTANT NOTE:")
    print("   Yahoo Finance doesn't provide historical options chains.")
    print("   This script uses TODAY's data + synthetic variation")
    print("   to approximate historical sentiment.")
    print("")
    print("   For production trading, consider:")
    print("   - Running this daily to build real history over time")
    print("   - Using paid data (CBOE, OptionMetrics, etc.)")
    print("="*60 + "\n")
    
    mode = input("Choose mode:\n  1. Create full history (first time)\n  2. Update today only (daily)\n\nChoice (1/2): ").strip()
    
    # Get list of valid tickers
    if not os.path.exists(RAW_STOCKS_DIR):
        print("❌ Stock data directory not found!")
        return
    
    files = [f for f in os.listdir(RAW_STOCKS_DIR) if f.endswith('_daily.csv')]
    tickers = [f.split('_')[0] for f in files]
    
    print(f"\nProcessing {len(tickers)} tickers...")
    
    successful = 0
    failed = 0
    
    # Process with progress bar
    for ticker in tqdm(tickers, desc="Options Data"):
        try:
            if mode == '1':
                # Full historical build
                if create_full_history_for_ticker(ticker):
                    successful += 1
                else:
                    failed += 1
            else:
                # Daily update
                if update_existing_ticker(ticker):
                    successful += 1
                else:
                    failed += 1
            
            # Rate limiting - Yahoo allows ~2000 requests/hour
            time.sleep(0.5)  # 2 requests/second = safe
            
        except Exception as e:
            failed += 1
            continue
    
    print(f"\n{'='*60}")
    print(f"COMPLETE!")
    print(f"  Success: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Data saved to: {OPTIONS_DIR}")
    print(f"{'='*60}\n")
    
    if mode == '1':
        print("TIP: Run this daily in mode 2 to build real historical data over time!")

if __name__ == "__main__":
    main()