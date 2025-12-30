import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
import random
import torch
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

# --- CONFIG ---
RAW_STOCKS_DIR = 'TrainingData/indicators_data/raw/stocksData'
SENTIMENT_DIR = 'TrainingData/indicators_data/raw/sentiment'

# Stealthy Headers (Looks like real Chrome)
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Upgrade-Insecure-Requests': '1',
    'Connection': 'keep-alive'
}

# --- LOAD FINBERT ---
print("Loading FinBERT Model...")
finbert_name = "yiyanghkust/finbert-tone"
tokenizer = BertTokenizer.from_pretrained(finbert_name)
model = BertForSequenceClassification.from_pretrained(finbert_name)
device = 0 if torch.cuda.is_available() else -1
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device)

def get_finbert_sentiment(ticker):
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    try:
        # Random sleep to look human (0.5 to 1.5 seconds)
        time.sleep(random.uniform(0.5, 1.5))
        
        response = requests.get(url, headers=HEADERS, timeout=10)
        
        # Check if we are blocked
        if response.status_code != 200:
            print(f"Blocked by Finviz for {ticker} (Status: {response.status_code})")
            return 0, 0
            
        soup = BeautifulSoup(response.text, 'html.parser')
        news_table = soup.find(id='news-table')
        
        if not news_table: 
            return 0, 0
        
        headlines = []
        for row in news_table.find_all('tr')[:10]:
            tag = row.find('a')
            if tag: headlines.append(tag.get_text().strip())

        if not headlines: return 0, 0
        
        # Analyze with FinBERT
        results = nlp(headlines)
        total = 0
        for res in results:
            if res['label'] == "Positive": total += res['score']
            elif res['label'] == "Negative": total -= res['score']
            
        return total / len(headlines), len(headlines)
    except Exception as e:
        print(f"Error {ticker}: {e}")
        return 0, 0

def main():
    if not os.path.exists(RAW_STOCKS_DIR):
        print(f"Error: {RAW_STOCKS_DIR} not found.")
        return

    files = [f for f in os.listdir(RAW_STOCKS_DIR) if f.endswith('_daily.csv')]
    total = len(files)
    print(f"Updating Sentiment (Stitching History + Live)...")

    for idx, filename in enumerate(files, 1):
        ticker = filename.split('_')[0]
        stock_path = os.path.join(RAW_STOCKS_DIR, filename)
        sentiment_path = os.path.join(SENTIMENT_DIR, f"{ticker}_sentiment_daily.csv")

        # 1. Get Live Sentiment (FinBERT)
        live_score, live_count = get_finbert_sentiment(ticker)
        
        # 2. Prepare the History DataFrame
        if os.path.exists(sentiment_path):
            try:
                # Load the Kaggle history we just made
                df_hist = pd.read_csv(sentiment_path)
                df_hist['date'] = pd.to_datetime(df_hist['date'])
            except:
                df_hist = pd.DataFrame(columns=['date', 'sentiment', 'num_articles'])
        else:
            df_hist = pd.DataFrame(columns=['date', 'sentiment', 'num_articles'])

        # 3. Get Master Date Range from Price Data
        df_stock = pd.read_csv(stock_path)
        if 'date' not in df_stock.columns: continue
        master_dates = pd.to_datetime(df_stock['date'])

        # 4. Create the Final DataFrame aligned to Stock Dates
        df_final = pd.DataFrame({'date': master_dates})
        
        # Merge history onto the master dates
        df_final = pd.merge(df_final, df_hist, on='date', how='left')
        
        # Fill missing history with Neutral (0.0)
        df_final['sentiment'] = df_final['sentiment'].fillna(0.0)
        df_final['num_articles'] = df_final['num_articles'].fillna(0)

        # 5. STITCH: Apply Live Score to the last 7 days
        if live_count > 0:
            df_final.iloc[-7:, df_final.columns.get_loc('sentiment')] = live_score
            df_final.iloc[-7:, df_final.columns.get_loc('num_articles')] = live_count
            print(f"[{idx}/{total}] {ticker}: History loaded + Live Score {live_score:.2f} ({live_count} articles)")
        else:
            if idx % 50 == 0: print(f"[{idx}/{total}] {ticker}: History loaded (No live news or Blocked)")

        # Save
        df_final.to_csv(sentiment_path, index=False)

    print("All Data Stitched!")

if __name__ == "__main__":
    main()