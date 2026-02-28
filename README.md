# LSTM AI Stock Predictor

> A multi-GPU deep learning system for predicting short-term stock movements across the US equity market. Models trained here are deployed live on **[SeekingBeta](https://github.com/enens-lab/SeekingBeta)**, a stock prediction web platform.

---

## Overview

Two LSTM models are trained on 3,000+ US stocks, engineered for **4x NVIDIA RTX 6000 Ada** (192 GB total VRAM):

| Model | Target | Horizon | Use Case |
|---|---|---|---|
| **Production** | > 2% return | 5 trading days | Consistent swing trades |
| **Jackpot** | > 20% return | 20 trading days | High-conviction breakout plays |

Both models output a probability score (0.0–1.0). Scores above the confidence threshold surface as picks.

---

## Architecture

### Model Design

```
Input: (batch_size, 60 timesteps, 49 features)
  │
  ├─ Conv1D(64, kernel=3, relu) + BatchNorm
  │
  ├─ LSTM(256, return_sequences=True) + BatchNorm + Dropout(0.3)
  │
  ├─ MultiHeadAttention(4 heads, key_dim=32) + Residual + LayerNorm
  │
  ├─ LSTM(128) + BatchNorm + Dropout(0.3)
  │
  ├─ Dense(32, relu)
  │
  └─ Dense(1, sigmoid) → Probability [float32]

Total Parameters: ~673K
```

Key design choices:
- **Multi-Head Attention** between LSTM layers captures long-range temporal dependencies
- **Warmup + Cosine Decay with Restarts** LR schedule
- **Temporal train/val split** at `2024-01-01` — model trains on 2021–2023 data, validates on 2024+ (unseen market regime)
- **Per-file scaler fitted on training rows only** — no future-statistics leakage
- **float32 precision** — numerical stability over float16 for this architecture
- **4-GPU distributed training** via `tf.distribute.MirroredStrategy`

### Feature Set (49 features)

| Category | Features | Count |
|---|---|---|
| Technical Indicators | RSI(14), MACD, Bollinger Bands, ATR | ~10 |
| Return Features | Log returns (close, volume), ROC | ~5 |
| Jackpot Indicators | BB Width (squeeze), RVOL, Distance to 52-week High | ~4 |
| Short Squeeze Score | Composite: short interest + momentum + volume + breakout | 1 |
| Multi-Timeframe Momentum | 3d, 10d, 30d, 90d momentum + acceleration | 5 |
| Volume Profile | Money Flow Multiplier, Accum/Distribution, Buy/Sell Pressure | 3 |
| Options Flow | Put/Call Ratio, UOA flag, Options Sentiment, Total Volume | 4 |
| Sector Encoding | One-hot encoded sector from Finviz | ~12 |
| Insider Trading | SEC Form 4 daily shares/amount/direction | ~3 |
| Sentiment | FinBERT news sentiment score + article count | 2 |
| Macro | VIX Fear Index | 1 |

> **Note:** Static Finviz fundamentals (P/E, short float, insider/institutional ownership, market cap) were removed from the feature set. They are scraped at a single point in time and stamped across all historical rows — a form of look-ahead bias. Sector dummies are retained since sector membership is stable.

---

## Project Structure

```
project_root/
├── train_model.py                  # Production 5-day model training (chunked, multi-GPU)
├── train_jackpot_model.py          # Jackpot 20-day model training (chunked, multi-GPU)
├── run_backtest_v4_1.py            # Backtest Engine v2 (SPY benchmark, slippage, risk metrics)
├── export_for_pythia.py            # Export models to SeekingBeta artifact format
├── daily_picks.py                  # Inference — scans market for top picks
├── master_pipeline.py              # End-to-end pipeline orchestrator
├── options_flow.py                 # Options flow data collection (put/call, UOA)
│
├── TrainingData/
│   ├── processor.py                # Feature engineering (OHLCV → 49 features)
│   ├── featuresPy/
│   │   ├── markets.py              # VIX/macro data fetcher
│   │   ├── stockScrapper.py        # OHLCV price data fetcher
│   │   ├── sentiment.py            # FinBERT news sentiment (date-aware, cached)
│   │   └── insiderbuying.py        # SEC Form 4 insider filings
│   ├── models/                     # Saved models + scalers (gitignored)
│   │   ├── lstm_production.keras
│   │   ├── lstm_jackpot.keras
│   │   ├── scaler_production.joblib
│   │   ├── scaler_jackpot.joblib
│   │   ├── feature_columns_production.json
│   │   └── feature_columns_jackpot.json
│   └── indicators_data/            # Raw + processed data (gitignored)
│       ├── raw/
│       │   ├── stocksData/         # OHLCV from Yahoo Finance
│       │   ├── optionsFlow/        # Options data (put/call, UOA)
│       │   ├── sentiment/          # FinBERT output
│       │   ├── insiderBuying/      # SEC filings
│       │   └── SPY-VIX/            # Market context
│       └── processed/
│           └── stocksData/         # 49-feature processed files
```

---

## Installation

### Dependencies

```bash
pip install -r requirements.txt
```

```
tensorflow>=2.13.0
pandas>=2.0.0
numpy>=1.24.0
pandas-ta>=0.3.14b
yfinance>=0.2.50
scikit-learn>=1.3.0
matplotlib>=3.7.0
tqdm>=4.65.0
joblib>=1.3.0
transformers>=4.30.0
requests>=2.28.0
beautifulsoup4>=4.11.0
```

> **yfinance >= 0.2.50 required.** Earlier versions have a `TypeError` bug in `yf.download()`. The backtest uses `Ticker.history()` to avoid this, but upgrading is still recommended.

### Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU | 1x NVIDIA GPU (8 GB VRAM) | 4x RTX 6000 Ada (48 GB each) |
| RAM | 32 GB | 64 GB+ |
| Storage | 50 GB | 200 GB (full dataset) |
| CUDA | 11.8+ | 12.x |

Training batch size auto-scales: `BATCH_SIZE_PER_REPLICA × num_GPUs`.

---

## Usage

### First-Time Setup

```bash
# 1. Collect raw data
python TrainingData/featuresPy/markets.py          # VIX, macro data
python TrainingData/featuresPy/stockScrapper.py    # OHLCV prices
python options_flow.py                             # Options data (choose mode 1: full history)
python TrainingData/featuresPy/sentiment.py        # News sentiment (requires Finviz Elite API token)

# 2. Process features (produces 49-feature dataset)
python TrainingData/processor.py

# 3. Train models
python train_model.py           # Production model (~2–3 hours on 4x GPU)
python train_jackpot_model.py   # Jackpot model (~2–3 hours on 4x GPU)

# 4. Backtest
python run_backtest_v4_1.py
```

### Daily Maintenance

```bash
# Update today's data
python options_flow.py                             # Choose mode 2: update today only
python TrainingData/featuresPy/sentiment.py        # Refresh news sentiment
python TrainingData/processor.py

# Retrain weekly/monthly — not required daily
```

### Run Inference

```bash
python daily_picks.py --mode consistency   # Conservative swing picks (production model)
python daily_picks.py --mode jackpot       # High-risk/reward picks (jackpot model)
```

### Full Pipeline (Data → Train → Inference)

```bash
python master_pipeline.py
```

---

## Data Sources

| Source | Data | Script |
|---|---|---|
| Yahoo Finance | OHLCV prices | `stockScrapper.py` |
| Yahoo Finance | Options chains (current) | `options_flow.py` |
| Finviz Elite | News headlines for sentiment | `sentiment.py` |
| FinBERT | Sentiment scoring | `sentiment.py` |
| SEC EDGAR | Insider trading (Form 4) | `insiderbuying.py` |
| Yahoo Finance | VIX fear index | `markets.py` |

---

## Training Details

### Key Parameters

| Parameter | Production | Jackpot |
|---|---|---|
| Prediction horizon | 5 days | 20 days |
| Target threshold | > 2% gain | > 20% gain |
| Sequence length | 60 days | 60 days |
| Batch size per GPU | 256 | 512 |
| Global batch (4 GPUs) | 1024 | 2048 |
| Initial LR | 5e-4 | 5e-4 |
| Warmup LR start | 1e-6 | 1e-6 |
| Gradient clipping | clipnorm=1.0 | clipnorm=1.0 |
| Early stopping patience | 15 epochs | 15 epochs |
| Chunk size | 400 files | 400 files |
| Max samples per chunk | 800,000 | 800,000 |
| Train/val cutoff | 2024-01-01 | 2024-01-01 |

### Temporal Validation Split

The old 90/10 random file split was replaced with a date-based split:

```
All stocks
    ├─ date < 2024-01-01  →  Training   (2021–2023 market regimes)
    └─ date >= 2024-01-01 →  Validation (2024–present, never seen during training)
```

This eliminates regime-leakage where the model could learn bull/bear patterns from the training period and validate on the same calendar windows in a different set of stocks. The `StandardScaler` is also fitted on pre-cutoff rows only so normalisation statistics don't carry future information.

### Data Pipeline Flow

```
1. Raw Data Collection (daily)
   ├─ stockScrapper.py      → OHLCV data
   ├─ options_flow.py       → Options metrics
   ├─ sentiment.py          → FinBERT scores (date-aware, incremental cache)
   ├─ insiderbuying.py      → SEC Form 4 filings
   └─ markets.py            → VIX, macro data

2. Feature Engineering (processor.py)
   └─ 49 features per stock per day

3. Training (weekly/monthly)
   ├─ train_model.py        → Production model
   └─ train_jackpot_model.py → Jackpot model

4. Backtesting (run_backtest_v4_1.py)
   └─ SPY-benchmarked simulation, trade logs, charts

5. Export to SeekingBeta (export_for_pythia.py)
   └─ Artifacts dropped into pythia_divination/artifacts/

6. Inference (daily_picks.py)
   └─ Ranked picks by confidence score
```

---

## Backtesting

`run_backtest_v4_1.py` — Backtest Engine v2 — runs a historical simulation with no look-ahead:

- **Multi-position portfolio:** 10 equal-weight slots, $10,000 starting capital
- **Tiered slippage:** 0.10% (liquid), 0.20% (medium), 0.50% (illiquid) per side based on 20-day avg volume
- **Liquidity filter:** price > $5/share, avg daily volume > 100,000 shares
- **SPY buy-and-hold benchmark** via `yfinance`, cached locally
- **Risk metrics:** Sharpe, Sortino, max drawdown, Calmar, alpha, beta, profit factor
- **Per-year breakdown** for walk-forward diagnostic
- **Confidence thresholds:** ≥ 0.60 (production), ≥ 0.55 (jackpot)
- Outputs: trade log CSV, metrics JSON, 3-panel performance chart → `backtest_results/`

### Latest Backtest Results (Production Model)

| Metric | Value |
|---|---|
| Total return (2024–2026) | +103.8% |
| CAGR | +43.2% |
| Sharpe ratio | 1.50 |
| Sortino ratio | 2.59 |
| Max drawdown | -13.8% |
| Calmar ratio | 3.12 |
| Win rate | 58.7% |
| Profit factor | 1.59x |
| Trades | 929 |

> These results use the production model trained on the old random file split and are provided as a reference baseline. Re-train with the temporal split to get validated out-of-sample metrics.

---

## SeekingBeta Integration

This repository's trained models power **[SeekingBeta](https://github.com/enens-lab/SeekingBeta)**, a stock prediction web platform built on the `pythia_divination` FastAPI backend.

### Artifact Export

```bash
python export_for_pythia.py --pythia-dir /path/to/SeekingBeta/pythia_divination
```

This copies the trained models into the directory structure that SeekingBeta's prediction server expects:

```
pythia_divination/artifacts/
├── lstm_5d/classifier/
│   ├── model.keras             ← trained production model
│   ├── feature_columns.json    ← locked 49-feature schema
│   ├── scaler.joblib           ← StandardScaler fitted on pre-2024 data
│   └── metrics.json            ← model metadata
└── lstm_jackpot/classifier/
    ├── model.keras
    ├── feature_columns.json
    ├── scaler.joblib
    └── metrics.json
```

### How Inference Works in SeekingBeta

SeekingBeta's `pythia_divination` server fetches raw OHLCV data for a ticker, then the LSTM model wrapper computes its own features via `compute_features()`:

```
SeekingBeta fetches OHLCV data for a ticker
        │
        ▼
    Model wrapper calls compute_features(OHLCV)
        │
        ├─ Derives technical features: RSI, MACD, BB, ATR, momentum, volume profile
        │  (all computable from OHLCV alone)
        │
        └─ External features (VIX, insider, options, sentiment) → default 0.0
               (technical features carry the majority of the signal)
        │
        ▼
    Scale → Sequence(60 days) → model.predict() → confidence score (0.0–1.0)
```

### API Endpoints (pythia_divination)

```bash
# List all available models
curl localhost:8000/models

# Get prediction from 5-day production model
curl "localhost:8000/predict/AAPL?model=lstm_5d"

# Get prediction from jackpot model
curl "localhost:8000/predict/AAPL?model=lstm_jackpot"

# Multi-model consensus
curl -X POST localhost:8000/predict/multi \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "models": ["lstm_5d", "lstm_jackpot"]}'
```

### Model Availability by Tier

| Tier | Models |
|---|---|
| Free | None |
| Pro | `lstm_5d`, `lstm_jackpot` |
| Enterprise | `lstm_5d`, `lstm_jackpot` |

### Compatibility Notes

- The `feature_columns.json` exported from this repo must match exactly what SeekingBeta's model wrapper uses at inference time. After any retraining, always re-run `export_for_pythia.py`.
- The `scaler.joblib` is fit on pre-2024 training data. SeekingBeta applies it to live OHLCV sequences — this is consistent with how the model was trained.
- External features (VIX, insider flow, options, sentiment) are zeroed at SeekingBeta inference time since they require separate data pipelines. The model was trained with these features present, so zeroing them degrades accuracy slightly but does not break inference.

---

## Troubleshooting

### Out-of-Memory During Training
Reduce `CHUNK_SIZE` in `train_model.py` / `train_jackpot_model.py` (default: 400 files per chunk). Each chunk is also hard-capped at 800,000 samples before being passed to `tf.data.Dataset.from_tensor_slices`, which pins data as a GPU tensor.

### Gradient Explosion
Gradient clipping (`clipnorm=1.0`) is enabled by default. If loss diverges, reduce the initial learning rate from `5e-4` to `1e-4`.

### Feature Count Mismatch at Inference
The `feature_columns.json` locks the feature schema at training time. If you re-run `processor.py` with changed features and retrain, always re-export to SeekingBeta with `export_for_pythia.py`. Mismatched schema will silently zero-fill missing columns.

### SPY Benchmark Missing in Backtest
`yf.download()` has a known `TypeError` bug in some versions. The backtest uses `Ticker.history()` instead. If SPY still fails, upgrade yfinance:
```bash
pip install --upgrade yfinance
```
SPY data is cached to `backtest_results/spy_cache.csv` after the first successful fetch.

### Sentiment Not Updating
`sentiment.py` caches headlines to `cache/headlines_dump.csv`. Delete this file to force a fresh fetch. Requires an active **Finviz Elite** subscription — the API token is set at the top of `sentiment.py`.

### Options Data Accuracy
The historical options backfill is synthetic (today's put/call ratio ± random walk). It provides a reasonable signal proxy but is not real historical data. Build a real time series by running `python options_flow.py` (mode 2) every trading day.

---

## Known Limitations

1. **Options history is approximate** — Yahoo Finance provides only the current options chain. True historical data requires a paid provider (CBOE DataShop, OptionMetrics).
2. **External features zeroed at SeekingBeta inference** — VIX, insider flow, options, and sentiment are not available in SeekingBeta's inference pipeline. Technical features carry the majority of the signal.
3. **Survivorship bias** — The training universe contains only stocks currently in the database. Delisted stocks are absent, which inflates historical positive rates slightly.
4. **Finviz Elite required for sentiment** — Free Finviz accounts do not provide the news table API used by `sentiment.py`.

---

## Future Enhancements

- Real historical options data integration (CBOE DataShop or OptionMetrics)
- Probability calibration (Platt scaling) to convert raw sigmoid scores to calibrated probabilities
- Monte Carlo Dropout for prediction uncertainty estimates
- Multi-task single model with shared encoder + two output heads (5d and 20d simultaneously)
- Temporal Fusion Transformer (TFT) architecture for improved long-range dependencies
- Ensemble of LSTM + gradient boosting for calibrated probabilities
- Intraday features (15m/1h bars for entry timing)

---

## License & Disclaimer

Research project — not financial advice. All trading involves risk. Past model performance does not guarantee future results. Use at your own risk.
