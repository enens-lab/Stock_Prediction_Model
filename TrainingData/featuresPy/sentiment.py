import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
import random
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm
import warnings

# --- CONFIG ---
API_TOKEN = "d4ffc539-7703-4e63-b5f4-2ffd1c0a1942" 

RAW_STOCKS_DIR = 'TrainingData/indicators_data/raw/stocksData'
SENTIMENT_DIR = 'TrainingData/indicators_data/raw/sentiment'
BASE_URL = "https://elite.finviz.com/quote.ashx"
BATCH_SIZE = 512 # Increased for T4/A100

# Silence Warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('future.no_silent_downcasting', True)

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

def get_device():
    if torch.cuda.is_available():
        print(f"GPU Detected: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    print("No GPU detected. Running on CPU (Slow).")
    return torch.device("cpu")

def fetch_headlines_batch(tickers):
    """
    Phase 1: Network Harvesting
    """
    collected_data = []
    
    with requests.Session() as session:
        session.headers.update(headers)
        
        # Shuffle tickers so we don't hammer 'A' stocks every time if we crash
        random.shuffle(tickers)
        
        for ticker in tqdm(tickers, desc="🌐 Harvesting Headlines"):
            url = f"{BASE_URL}?t={ticker}&auth={API_TOKEN}"
            try:
                # Faster sleep (0.05 is safe for Elite API usually)
                time.sleep(random.uniform(0.01, 0.1)) 
                
                resp = session.get(url, timeout=5)
                if resp.status_code != 200: continue

                soup = BeautifulSoup(resp.text, 'html.parser')
                news_table = soup.find(id='news-table')
                if not news_table: continue

                for row in news_table.find_all('tr')[:50]:
                    tag = row.find('a')
                    if tag:
                        collected_data.append({
                            'ticker': ticker,
                            'headline': tag.get_text().strip(),
                        })
            except:
                continue
                
    return pd.DataFrame(collected_data)

def run_inference(df, model, tokenizer, device):
    """
    Phase 2: GPU Mass Inference
    """
    if df.empty: return df
    
    print(f"Analyzing {len(df)} headlines on GPU...")
    headlines = df['headline'].tolist()
    scores = []
    
    model.eval()
    
    with torch.no_grad():
        for i in tqdm(range(0, len(headlines), BATCH_SIZE), desc="🔥 GPU Firing"):
            batch_texts = headlines[i:i+BATCH_SIZE]
            
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=64)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            label2id = model.config.label2id
            pos_id = label2id.get('Positive', 1)
            neg_id = label2id.get('Negative', 2)
            
            batch_scores = (probs[:, pos_id] - probs[:, neg_id]).cpu().numpy()
            scores.extend(batch_scores)
            
    df['score'] = scores
    return df

def main():
    if "PASTE" in API_TOKEN:
        print("STOP: Update API_TOKEN in the script first!")
        return

    # 1. Setup
    device = get_device()
    finbert_name = "yiyanghkust/finbert-tone"
    tokenizer = BertTokenizer.from_pretrained(finbert_name)
    model = BertForSequenceClassification.from_pretrained(finbert_name).to(device)

    # 2. Get Tickers
    if not os.path.exists(RAW_STOCKS_DIR):
        print("Raw stocks directory not found.")
        return
        
    all_files = [f for f in os.listdir(RAW_STOCKS_DIR) if f.endswith('_daily.csv')]
    all_tickers = [f.split('_')[0] for f in all_files]
    print(f"Job: Scan {len(all_tickers)} stocks.")

    # 3. HARVEST (Network Bound)
    cache_path = "cache/headlines_dump.csv"
    if os.path.exists(cache_path):
        print("Found cached headlines! Loading from disk to save time...")
        df_headlines = pd.read_csv(cache_path)
    else:
        df_headlines = fetch_headlines_batch(all_tickers)
        os.makedirs("cache", exist_ok=True)
        df_headlines.to_csv(cache_path, index=False)
        print(f"Saved backup to {cache_path}")
    
    print(f"Processing {len(df_headlines)} headlines.")
    
    if df_headlines.empty:
        print("No headlines found.")
        return

    # 4. INFERENCE (GPU Bound)
    if 'score' not in df_headlines.columns:
        df_scored = run_inference(df_headlines, model, tokenizer, device)
    else:
        df_scored = df_headlines 

    # 5. AGGREGATE & SMART UPDATE (Disk Bound - OPTIMIZED)
    print("Aggregating Scores...")
    
    ticker_scores = df_scored.groupby('ticker')['score'].mean()
    ticker_counts = df_scored.groupby('ticker')['score'].count()

    os.makedirs(SENTIMENT_DIR, exist_ok=True)

    # --- THE OPTIMIZATION ---
    # Only process stocks that actually HAVE news today.
    # The other 6000 stocks will be handled by processor.py (fillna=0)
    active_tickers = list(ticker_scores.index)
    
    # Filter active tickers to only those that match valid stock files
    # (Avoids creating dummy files for tickers not in our stock database)
    valid_active_tickers = [t for t in active_tickers if t in all_tickers]
    
    print(f"⚡ FAST UPDATE: Updating only {len(valid_active_tickers)} stocks with news (Skipping {len(all_tickers) - len(valid_active_tickers)} silent stocks)...")

    for ticker in tqdm(valid_active_tickers, desc="Stitching History"):
        save_path = os.path.join(SENTIMENT_DIR, f"{ticker}_sentiment_daily.csv")
        stock_path = os.path.join(RAW_STOCKS_DIR, f"{ticker}_daily.csv")
        
        # Load Price Data for Date Index
        try:
            df_stock = pd.read_csv(stock_path)
            if 'date' not in df_stock.columns: continue
            df_stock['date'] = pd.to_datetime(df_stock['date'])
        except:
            continue

        # Load Existing History OR Create New
        if os.path.exists(save_path):
            try:
                df_hist = pd.read_csv(save_path)
                df_hist['date'] = pd.to_datetime(df_hist['date'])
            except:
                df_hist = pd.DataFrame(columns=['date', 'sentiment', 'num_articles'])
        else:
            df_hist = pd.DataFrame(columns=['date', 'sentiment', 'num_articles'])

        # Create the Base Frame
        df_final = pd.DataFrame({'date': df_stock['date']})
        
        # Merge Old History
        df_final = pd.merge(df_final, df_hist, on='date', how='left')
        
        # Fill NaNs
        df_final['sentiment'] = df_final['sentiment'].fillna(0.0)
        df_final['num_articles'] = df_final['num_articles'].fillna(0)

        # Apply Live Score
        live_score = ticker_scores[ticker]
        live_count = ticker_counts[ticker]
        
        # Update last 5 days
        last_5_indices = df_final.index[-5:]
        df_final.loc[last_5_indices, 'sentiment'] = live_score
        df_final.loc[last_5_indices, 'num_articles'] = live_count

        # Save
        df_final.to_csv(save_path, index=False)

    print("COMPLETE. Sentiment history updated.")

if __name__ == "__main__":
    main()