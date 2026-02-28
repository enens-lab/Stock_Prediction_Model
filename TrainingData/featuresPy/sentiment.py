import re
import requests
import logging
from bs4 import BeautifulSoup
from datetime import datetime, date
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
CACHE_PATH = "cache/headlines_dump.csv"
BATCH_SIZE = 512

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('future.no_silent_downcasting', True)

logging.basicConfig(
    filename='sentiment_errors.log',
    level=logging.WARNING,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}


def get_device():
    if torch.cuda.is_available():
        print(f"GPU Detected: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    print("No GPU detected. Running on CPU (Slow).")
    return torch.device("cpu")


def parse_finviz_date(text, last_date):
    """
    Parse the date/time cell from a Finviz news table row.

    Finviz format:
      Full row (new day): 'Feb-26-26 10:30AM'
      Time-only (same day): ' 09:15AM'

    Returns the parsed date, or last_date when only a time is present.
    """
    text = text.strip()
    m = re.match(r'([A-Za-z]{3}-\d{1,2}-\d{2,4})', text)
    if m:
        date_part = m.group(1)
        for fmt in ('%b-%d-%y', '%b-%d-%Y'):
            try:
                return datetime.strptime(date_part, fmt).date()
            except ValueError:
                continue
    # Time-only row — inherit the date from the previous row
    return last_date


def fetch_headlines_batch(tickers):
    """
    Phase 1: Network Harvesting.

    Extracts (ticker, date, headline) from Finviz Elite's news table.
    The date column is new: Finviz shows a date on the first article of each
    calendar day and only a time on subsequent same-day articles; we carry the
    date forward across those rows.
    """
    collected_data = []

    with requests.Session() as session:
        session.headers.update(headers)
        random.shuffle(tickers)

        for ticker in tqdm(tickers, desc="Harvesting Headlines"):
            url = f"{BASE_URL}?t={ticker}&auth={API_TOKEN}"
            try:
                time.sleep(random.uniform(0.01, 0.1))
                resp = session.get(url, timeout=5)
                if resp.status_code != 200:
                    logging.warning(f"{ticker}: HTTP {resp.status_code}")
                    continue

                soup = BeautifulSoup(resp.text, 'html.parser')
                news_table = soup.find(id='news-table')
                if not news_table:
                    continue

                current_date = date.today()
                for row in news_table.find_all('tr')[:50]:
                    cells = row.find_all('td')
                    if len(cells) < 2:
                        continue

                    current_date = parse_finviz_date(cells[0].get_text(), current_date)

                    link = cells[1].find('a')
                    if link:
                        collected_data.append({
                            'ticker': ticker,
                            'date': current_date,
                            'headline': link.get_text().strip(),
                        })

            except Exception as e:
                logging.warning(f"{ticker}: fetch error — {e}")
                continue

    return pd.DataFrame(collected_data)


def run_inference(df, model, tokenizer, device):
    """
    Phase 2: GPU Mass Inference.
    Adds a 'score' column (positive − negative softmax probability).
    """
    if df.empty:
        return df

    print(f"Analyzing {len(df)} headlines on GPU...")
    headlines = df['headline'].tolist()
    scores = []

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(headlines), BATCH_SIZE), desc="GPU Firing"):
            batch_texts = headlines[i:i + BATCH_SIZE]
            inputs = tokenizer(
                batch_texts, return_tensors="pt",
                padding=True, truncation=True, max_length=64,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

            label2id = model.config.label2id
            pos_id = label2id.get('Positive', 1)
            neg_id = label2id.get('Negative', 2)

            batch_scores = (probs[:, pos_id] - probs[:, neg_id]).cpu().numpy()
            scores.extend(batch_scores)

    df = df.copy()
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
    all_tickers_set = set(all_tickers)
    print(f"Job: Scan {len(all_tickers)} stocks.")

    # 3. HARVEST (Network Bound)
    os.makedirs("cache", exist_ok=True)
    if os.path.exists(CACHE_PATH):
        print("Found cached headlines! Loading from disk to save time...")
        df_headlines = pd.read_csv(CACHE_PATH)
        if 'date' in df_headlines.columns:
            df_headlines['date'] = pd.to_datetime(df_headlines['date']).dt.date
    else:
        df_headlines = fetch_headlines_batch(all_tickers)
        # Save raw headlines immediately (checkpoint before GPU work)
        df_headlines.to_csv(CACHE_PATH, index=False)
        print(f"Saved {len(df_headlines)} headlines to cache.")

    print(f"Processing {len(df_headlines)} headlines.")

    if df_headlines.empty:
        print("No headlines found. Check Finviz subscription / API token.")
        return

    # 4. INFERENCE (GPU Bound)
    # Skip if scores already present (e.g. cache was saved after a completed run)
    if 'score' not in df_headlines.columns:
        df_scored = run_inference(df_headlines, model, tokenizer, device)
        # Save scored cache so future runs skip FinBERT entirely
        df_scored.to_csv(CACHE_PATH, index=False)
        print("Scored cache saved.")
    else:
        df_scored = df_headlines
        print(f"Using pre-scored cache ({len(df_scored)} headlines, FinBERT skipped).")

    # 5. AGGREGATE per (ticker, date)
    # Old cache format had no date column — fall back gracefully
    print("Aggregating scores by ticker and date...")
    if 'date' not in df_scored.columns:
        print("  Warning: cache has no date column (old format). Assigning today's date.")
        df_scored = df_scored.copy()
        df_scored['date'] = pd.Timestamp.today().normalize()

    df_scored['date'] = pd.to_datetime(df_scored['date'])

    daily_agg = (
        df_scored
        .groupby(['ticker', 'date'])
        .agg(sentiment=('score', 'mean'), num_articles=('score', 'count'))
        .reset_index()
    )

    # 6. SMART UPDATE — only tickers with news
    active_tickers = [t for t in daily_agg['ticker'].unique() if t in all_tickers_set]
    print(f"Updating {len(active_tickers)} stocks with news "
          f"(skipping {len(all_tickers) - len(active_tickers)} silent stocks)...")

    os.makedirs(SENTIMENT_DIR, exist_ok=True)

    for ticker in tqdm(active_tickers, desc="Stitching History"):
        save_path = os.path.join(SENTIMENT_DIR, f"{ticker}_sentiment_daily.csv")
        stock_path = os.path.join(RAW_STOCKS_DIR, f"{ticker}_daily.csv")

        try:
            df_stock = pd.read_csv(stock_path, usecols=['date'])
            df_stock['date'] = pd.to_datetime(df_stock['date'])
        except Exception as e:
            logging.warning(f"{ticker}: could not load stock data — {e}")
            continue

        # Load or create existing sentiment history
        if os.path.exists(save_path):
            try:
                df_hist = pd.read_csv(save_path)
                df_hist['date'] = pd.to_datetime(df_hist['date'])
            except Exception:
                df_hist = pd.DataFrame(columns=['date', 'sentiment', 'num_articles'])
        else:
            df_hist = pd.DataFrame(columns=['date', 'sentiment', 'num_articles'])

        # New scores for this ticker
        new_scores = (
            daily_agg[daily_agg['ticker'] == ticker][['date', 'sentiment', 'num_articles']]
            .copy()
        )
        new_scores['date'] = pd.to_datetime(new_scores['date'])

        # Merge history with new scores; new scores win on overlap
        df_merged = pd.merge(df_hist, new_scores, on='date', how='outer', suffixes=('_old', '_new'))
        df_merged['sentiment'] = df_merged['sentiment_new'].combine_first(df_merged['sentiment_old'])
        df_merged['num_articles'] = df_merged['num_articles_new'].combine_first(df_merged['num_articles_old'])
        df_merged = df_merged[['date', 'sentiment', 'num_articles']]

        # Reindex to the stock's full date range, fill missing dates with 0
        df_final = pd.merge(df_stock, df_merged, on='date', how='left')
        df_final['sentiment'] = df_final['sentiment'].fillna(0.0)
        df_final['num_articles'] = df_final['num_articles'].fillna(0)

        df_final.to_csv(save_path, index=False)

    print("COMPLETE. Sentiment history updated.")


if __name__ == "__main__":
    main()
