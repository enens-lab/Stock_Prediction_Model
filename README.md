# LSTM AI Stock Predictor

> A multi-GPU deep learning system for predicting short-term stock movements across the entire US equity market, with integration into the [Pythia](https://github.com/aj-robb/pythia) trading platform.

## Overview

This system trains two LSTM (Long Short-Term Memory) models on ~2,000+ US stocks simultaneously, engineered for **4x NVIDIA ADA6000 GPUs** (192GB total VRAM):

| Model | Target | Horizon | Use Case |
|-------|--------|---------|----------|
| **Production (5-Day)** | >2% return | 5 trading days | Consistent swing trades |
| **Jackpot** | >20% return | 20 trading days | High-conviction breakout plays |

Both models output a probability score (0.0–1.0) for binary classification: will the stock hit the target return within the horizon?

---

## Architecture

### Model Design

```
Input (60 timesteps x ~50 features)
  │
  ├─ Conv1D(64, kernel=3) + BatchNorm
  │
  ├─ LSTM(256, return_sequences=True) + BatchNorm + Dropout(0.3)
  │
  ├─ MultiHeadAttention(4 heads, key_dim=32) + Residual + LayerNorm
  │
  ├─ LSTM(128) + BatchNorm + Dropout(0.3)
  │
  ├─ Dense(32, relu)
  │
  └─ Dense(1, sigmoid) → Probability
```

Key design choices:
- **Multi-Head Attention** between LSTM layers captures long-range temporal dependencies
- **Cosine Decay with Warm Restarts** learning rate schedule (1-epoch linear warmup)
- **File-based train/val split** (90/10) prevents data leakage between stocks
- **Mixed precision (float16)** training with float32 output layer for numerical stability
- **4-GPU distributed training** via `tf.distribute.MirroredStrategy`

### Feature Set (~50 features)

| Category | Features |
|----------|----------|
| **Technicals** | RSI(14), MACD, Bollinger Bands, ATR |
| **Returns** | Log returns (close, volume) |
| **Jackpot Indicators** | BB Width (squeeze), RVOL, Distance to 52-week High, ROC(10) |
| **Fundamentals** | P/E Ratio, Short Float, Insider/Institutional Ownership, Market Cap |
| **Sector** | One-hot encoded sector from Finviz |
| **Insider Trading** | SEC Form 4 daily shares/amount/direction |
| **Sentiment** | FinBERT news sentiment scores |
| **Macro** | VIX Fear Index |

---

## Project Structure

```
├── train_model.py              # Production 5-day model training
├── train_jackpot_model.py      # Jackpot 20-day model training
├── daily_picks.py              # Inference — scans market for top picks
├── master_pipeline.py          # End-to-end pipeline orchestrator
├── export_for_pythia.py        # Export models to Pythia artifact format
│
├── TrainingData/
│   ├── processor.py            # Feature engineering (OHLCV → 50 features)
│   ├── featuresPy/
│   │   ├── markets.py          # VIX/macro data fetcher
│   │   ├── stockScrapper.py    # OHLCV price data fetcher
│   │   └── sentiment.py        # FinBERT news sentiment
│   ├── models/                 # Saved models + scalers (gitignored)
│   │   ├── lstm_production.keras
│   │   ├── lstm_jackpot.keras
│   │   ├── scaler_production.joblib
│   │   ├── scaler_jackpot.joblib
│   │   ├── feature_columns_production.json
│   │   └── feature_columns_jackpot.json
│   └── indicators_data/        # Raw + processed data (gitignored)
```

---

## Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Collect Data

```bash
python TrainingData/featuresPy/markets.py        # VIX data
python TrainingData/featuresPy/stockScrapper.py   # OHLCV prices
python TrainingData/featuresPy/sentiment.py       # News sentiment
```

### 3. Process Features

```bash
python TrainingData/processor.py
```

### 4. Train Models

```bash
python train_model.py           # Production model (~2% in 5 days)
python train_jackpot_model.py   # Jackpot model (~20% in 20 days)
```

### 5. Run Inference

```bash
python daily_picks.py --mode consistency   # Conservative picks
python daily_picks.py --mode jackpot       # High-risk/reward picks
```

### 6. Full Pipeline (Data → Train → Inference → Trade)

```bash
python master_pipeline.py
```

---

## Pythia Integration

This model integrates with [Pythia](https://github.com/aj-robb/pythia), a FastAPI-based trading platform with a web UI. After integration, the LSTM models appear as selectable models in Pythia's existing interface — **no frontend changes required**.

### How It Works

Pythia normally computes 10 simple technical features for its built-in models. The LSTM models need ~50 features. The solution: each LSTM model wrapper has a `compute_features()` method that **derives its own features from raw OHLCV data** at inference time.

```
Pythia fetches OHLCV data for a ticker
        │
        ▼
    ┌──────────────────────┐
    │  Model has            │
    │  compute_features()?  │
    └──────┬───────┬───────┘
           │       │
       YES │       │ NO (legacy models)
           ▼       ▼
    ┌─────────┐  ┌──────────┐
    │ Compute  │  │ Pythia's │
    │ ~50 feat │  │ 10 feat  │
    │ from     │  │ pipeline │
    │ OHLCV    │  │          │
    └────┬─────┘  └────┬─────┘
         │             │
         ▼             ▼
      Scale → Sequence(60) → model.predict() → PredictionResult
```

Features derivable from OHLCV (RSI, MACD, BB, ATR, jackpot indicators, log returns) are computed directly. Features that require external data (fundamentals, insider, VIX) default to 0 — the model still performs well since the technical features carry most of the signal.

### Export Models to Pythia

After training, run the export script to copy model artifacts into Pythia's directory structure:

```bash
python export_for_pythia.py --pythia-dir /path/to/pythia_divination
```

This creates:

```
pythia_divination/artifacts/
├── lstm_5d/classifier/
│   ├── model.keras
│   ├── feature_columns.json
│   ├── scaler.joblib
│   └── metrics.json
└── lstm_jackpot/classifier/
    ├── model.keras
    ├── feature_columns.json
    ├── scaler.joblib
    └── metrics.json
```

### Pythia API Usage

Once exported, the models are available through Pythia's standard API:

```bash
# List all models (lstm_5d and lstm_jackpot should appear)
curl localhost:8000/models

# Get prediction from 5-day model
curl "localhost:8000/predict/AAPL?model=lstm_5d"

# Get prediction from jackpot model
curl "localhost:8000/predict/AAPL?model=lstm_jackpot"

# Multi-model consensus
curl -X POST localhost:8000/predict/multi \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "models": ["lstm_5d", "lstm_jackpot", "gradient_boosting"]}'
```

### Pythia Files Modified

The integration requires minimal changes to Pythia (on the `feature/lstm-stock-predictor-integration` branch):

| File | Change |
|------|--------|
| `models/` (new module) | Model registry, base classes, and wrappers for all model types |
| `models/lstm_stock_predictor.py` | LSTM wrapper with `compute_features()` for OHLCV-based feature computation |
| `api/service.py` | 3-line addition: check for `compute_features()` method before prediction |
| `api/tiers.py` | Added `lstm_5d` and `lstm_jackpot` to pro/enterprise allowed models |

### Tier Access

| Tier | LSTM Models Available |
|------|----------------------|
| Free | None |
| Pro | `lstm_5d`, `lstm_jackpot` |
| Enterprise | `lstm_5d`, `lstm_jackpot` |

---

Training batch size auto-scales: `BATCH_SIZE_PER_REPLICA * num_GPUs`.

---

## License

Research project — not financial advice. Use at your own risk.
