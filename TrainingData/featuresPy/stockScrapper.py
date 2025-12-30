import yfinance as yf
import pandas as pd
import os
import time

def download_yahoo_daily(tickers, save_folder="TrainingData/indicators_data/raw/stocksData"):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    print(f"Starting download for {len(tickers)} tickers using Yahoo Finance...")
    
    batch_size = 50
    
    for i in range(0, len(tickers), batch_size):
        raw_batch = tickers[i:i+batch_size]
        print(f"Processing batch {i} to {i+len(raw_batch)}...")
        
        # --- THE FIX ---
        # 1. Clean whitespace/uppercase
        # 2. Replace '/' with '-' (BRK/B -> BRK-B)
        # 3. Replace '.' with '-' (BRK.B -> BRK-B) <--- CRITICAL FIX FOR YAHOO
        batch = []
        for t in raw_batch:
            clean_t = t.strip().upper().replace('/', '-').replace('.', '-')
            if clean_t:
                batch.append(clean_t)
        
        if not batch: continue

        try:
            # Added auto_adjust=True (Fixes the Warning AND handles stock splits better)
            data = yf.download(batch, period="max", group_by='ticker', 
                               threads=True, progress=False, auto_adjust=True)
        except Exception as e:
            print(f"   Batch failed: {e}")
            continue

        for ticker in batch:
            try:
                # Extract dataframe
                if len(batch) > 1:
                    # Check if ticker exists in columns (handling yfinance multi-index weirdness)
                    if ticker not in data.columns.get_level_values(0):
                        print(f"   No data found for {ticker}")
                        continue
                    df = data[ticker].copy()
                else:
                    df = data.copy()

                if df.empty:
                    continue

                # Clean columns
                df.reset_index(inplace=True)
                df.columns = [c.lower() for c in df.columns]
                
                # Standardize Date
                if 'date' not in df.columns and 'datetime' not in df.columns:
                     for col in df.columns:
                         if 'date' in col.lower():
                             df.rename(columns={col: 'date'}, inplace=True)
                             break
                
                # Ensure we have a date column
                if 'date' not in df.columns:
                    continue

                # Keep required columns
                # Note: With auto_adjust=True, 'adj close' becomes 'close' automatically
                required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
                available_cols = [c for c in required_cols if c in df.columns]
                df = df[available_cols]

                # Format Date
                df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
                df.sort_values('date', inplace=True)
                
                # Remove rows with NaN in critical columns
                df.dropna(subset=['close'], inplace=True)

                # Save
                save_path = os.path.join(save_folder, f"{ticker}_daily.csv")
                df.to_csv(save_path, index=False)

            except Exception as e:
                # print(f"Error processing {ticker}: {e}") # Optional: uncomment for verbose debug
                pass
        
        # Polite sleep to avoid rate limits
        time.sleep(1)

    print("All downloads complete!")

if __name__ == "__main__":
    list_path = os.path.join('TrainingData', 'stockList.csv')
    
    if os.path.exists(list_path):
        with open(list_path, 'r') as file:
            # Read and filter empty lines
            tickers = [line.strip() for line in file if line.strip()]
        
        if tickers:
            download_yahoo_daily(tickers)
        else:
            print("Stock list is empty.")
    else:
        print(f"ERROR: Could not find {list_path}")