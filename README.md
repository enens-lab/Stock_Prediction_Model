# AI-Powered Stock Market Predictor (LSTM + FinBERT)

> **A high-performance quantitative trading engine utilizing Deep Learning to predict 5-day price movements across the entire US Stock Market.**

## System Overview
Unlike standard tutorials that train on a single stock, this system is engineered to process **2,000+ tickers simultaneously** using a vectorized pipeline optimized for NVIDIA A100 GPUs. It combines technical analysis, sentiment analysis, and insider trading data to generate probability-based buy signals.

### Key Capabilities
* **Scale:** Trains on the full Russell 2000 + S&P 500 (~2,156 stocks).
* **Architecture:** Deep LSTM (Long Short-Term Memory) Network for time-series classification.
* **Alternative Data:** Integrates **FinBERT** (NLP for news sentiment) and **SEC Form 4** (Insider Trading) data.
* **Performance:** Capable of identifying Sector Rotation (e.g., Tech to Consumer Staples) during market volatility.

---

## How It Works
The model treats the market as a **Binary Classification** problem:
* **Input:** 60-day lookback window of Price, Volume, RSI, MACD, Bollinger Bands, and Sentiment.
* **Target:** Will the stock price be **> 2% higher** in exactly **5 days**?
* **Output:** A confidence score (0.0 to 1.0). Trades are only taken when Confidence > 80%.

## Tech Stack
* **Core:** Python 3.10+, TensorFlow/Keras 2.x
* **Data Processing:** Pandas, NumPy (Vectorized operations for 30GB+ datasets)
* **NLP:** HuggingFace Transformers (FinBERT)
* **Hardware:** Optimized for CUDA (NVIDIA T4 / A100)

## Project Structure
* `daily_signals.py` - The "Production" script. Scans the market and prints top picks for tomorrow.
* `processor.py` - The ETL Engine. Calculates 50+ technical indicators and handles data cleaning.
* `lstm_model.h5` - The pre-trained model weights (GitIgnored due to size).
* `run_backtest_v4.py` - Simulation engine to verify historical performance.

## Usage
1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Update Data:**
    ```bash
    python TrainingData/featuresPy/stockScrapper.py
    python TrainingData/processor.py
    ```
3.  **Get Trading Signals:**
    ```bash
    python daily_signals.py
    ```

## Performance Note
In historical backtesting (2020-2025), the "High Confidence" strategy demonstrated a capability to outperform the SPY benchmark by filtering for high-momentum setups and rotating sectors during downturns.

---
*Disclaimer: This is an algorithmic research project. Not financial advice.*
