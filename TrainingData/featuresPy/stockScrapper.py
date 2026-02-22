import yfinance as yf
import pandas as pd
import os
import time

def download_yahoo_daily(tickers, save_folder="TrainingData/indicators_data/raw/stocksData"):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    print(f"Starting Smart Update for {len(tickers)} tickers...")
    
    batch_size = 50
    
    for i in range(0, len(tickers), batch_size):
        raw_batch = tickers[i:i+batch_size]
        print(f"Processing batch {i} to {i+len(raw_batch)}...")
        
        # 1. Clean Tickers
        batch_clean = []
        for t in raw_batch:
            clean_t = t.strip().upper().replace('/', '-').replace('.', '-')
            if clean_t:
                batch_clean.append(clean_t)
        
        if not batch_clean: continue

        # 2. SPLIT: Who is New? Who is Existing?
        # We process them separately to optimize speed.
        existing_tickers = []
        new_tickers = []
        
        for t in batch_clean:
            file_path = os.path.join(save_folder, f"{t}_daily.csv")
            if os.path.exists(file_path):
                existing_tickers.append(t)
            else:
                new_tickers.append(t)

        # --- PROCESS EXISTING (Update Mode) ---
        if existing_tickers:
            try:
                # We download 1y buffer to be safe (covers weekends/holidays/gaps)
                # This is much faster than 'max'
                data = yf.download(existing_tickers, period="1y", group_by='ticker', 
                                   threads=True, progress=False, auto_adjust=True)
                process_batch_data(data, existing_tickers, save_folder, is_update=True)
            except Exception as e:
                print(f"Update Batch failed: {e}")

        # --- PROCESS NEW (Full History Mode) ---
        if new_tickers:
            print(f"   Found {len(new_tickers)} new tickers. Downloading full history...")
            try:
                data = yf.download(new_tickers, period="max", group_by='ticker', 
                                   threads=True, progress=False, auto_adjust=True)
                process_batch_data(data, new_tickers, save_folder, is_update=False)
            except Exception as e:
                print(f"New Ticker Batch failed: {e}")

        # Polite sleep
        time.sleep(1)

    print("All downloads complete!")

def process_batch_data(data, tickers, save_folder, is_update):
    """
    Handles the cleaning, merging, and saving logic.
    """
    for ticker in tickers:
        try:
            # EXTRACT DATAFRAME
            if len(tickers) > 1:
                # Handle MultiIndex
                if ticker not in data.columns.get_level_values(0):
                    continue
                df_new = data[ticker].copy()
            else:
                df_new = data.copy()

            if df_new.empty: continue

            # CLEANUP
            df_new.reset_index(inplace=True)
            df_new.columns = [c.lower() for c in df_new.columns]
            
            # Standardize Date Column Name
            if 'date' not in df_new.columns:
                for col in df_new.columns:
                    if 'date' in col.lower():
                        df_new.rename(columns={col: 'date'}, inplace=True)
                        break
            
            if 'date' not in df_new.columns: continue

            # Filter Cols (STRICT FILTERING)
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            # Only keep columns that actually exist in new data
            final_cols = [c for c in required_cols if c in df_new.columns]
            df_new = df_new[final_cols]

            # Format Date
            df_new['date'] = pd.to_datetime(df_new['date']).dt.strftime('%Y-%m-%d')
            df_new.dropna(subset=['close'], inplace=True)

            # SAVE / MERGE LOGIC
            save_path = os.path.join(save_folder, f"{ticker}_daily.csv")
            
            if is_update and os.path.exists(save_path):
                # Load Old
                df_old = pd.read_csv(save_path)
                
                # --- FIX FOR WARNING ---
                # Ensure Old DF has exactly the same columns as New DF
                # This prevents 'FutureWarning' about mismatching columns
                df_old = df_old[final_cols]
                
                # Concat Old + New (Now they match perfectly)
                df_final = pd.concat([df_old, df_new])
                
                # Deduplicate
                df_final.drop_duplicates(subset='date', keep='last', inplace=True)
                df_final.sort_values('date', inplace=True)
                
                # Save
                df_final.to_csv(save_path, index=False)
            else:
                # Just Save
                df_new.sort_values('date', inplace=True)
                df_new.to_csv(save_path, index=False)

        except Exception as e:
            # print(f"Error on {ticker}: {e}")
            pass

if __name__ == "__main__":
    list_path = os.path.join('TrainingData', 'stockList.csv')
    
    if os.path.exists(list_path):
        with open(list_path, 'r') as file:
            tickers = [line.strip() for line in file if line.strip()]
        
        if tickers:
            download_yahoo_daily(tickers)
        else:
            print("Stock list is empty.")
    else:
        print(f"ERROR: Could not find {list_path}")