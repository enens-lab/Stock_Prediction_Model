"""
Backtest Script for LSTM Models (Production + Jackpot)

Compatible with:
- Production Model: Predicts >2% gain in 5 days
- Jackpot Model: Predicts >20% gain in 20 days

Features:
- Loads models and their feature maps
- Runs historical simulation
- Compares to SPY benchmark
- Generates performance charts
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import json
import joblib
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

# --- CUSTOM LEARNING RATE SCHEDULE (Must be defined before loading models) ---
@keras.utils.register_keras_serializable(package="Custom", name="WarmupCosineDecay")
class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    """Custom learning rate schedule with warmup and cosine decay."""
    
    def __init__(self, warmup_steps, total_decay_steps=None, initial_lr=1e-3, 
                 warmup_start_lr=1e-5, t_mul=2.0, m_mul=0.9, **kwargs):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        self.warmup_start_lr = warmup_start_lr
        
        # Handle legacy models that might not have total_decay_steps
        if total_decay_steps is None:
            total_decay_steps = warmup_steps * 5
            
        self.cosine_schedule = keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=initial_lr,
            first_decay_steps=total_decay_steps,
            t_mul=t_mul,
            m_mul=m_mul,
        )

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)

        warmup_lr = self.warmup_start_lr + (self.initial_lr - self.warmup_start_lr) * (step / warmup_steps)
        cosine_lr = self.cosine_schedule(step - warmup_steps)

        return tf.where(step < warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "warmup_steps": self.warmup_steps,
            "initial_lr": self.initial_lr,
            "warmup_start_lr": self.warmup_start_lr,
        }

# --- CONFIG ---
PROJECT_ROOT = Path(os.getcwd())
PROCESSED_DIR = PROJECT_ROOT / "TrainingData/indicators_data/processed/stocksData"
SPY_PATH = PROJECT_ROOT / "TrainingData/indicators_data/processed/SPY-VIX/SPY_daily_processed.csv"
OUTPUT_DIR = PROJECT_ROOT / "backtest_results"

# Model Paths
PRODUCTION_MODEL = PROJECT_ROOT / "TrainingData/models/lstm_production.keras"
PRODUCTION_FEATURES = PROJECT_ROOT / "TrainingData/models/feature_columns_production.json"
PRODUCTION_SCALER = PROJECT_ROOT / "TrainingData/models/scaler_production.joblib"

JACKPOT_MODEL = PROJECT_ROOT / "TrainingData/models/lstm_jackpot.keras"
JACKPOT_FEATURES = PROJECT_ROOT / "TrainingData/models/feature_columns_jackpot.json"
JACKPOT_SCALER = PROJECT_ROOT / "TrainingData/models/scaler_jackpot.joblib"

# Strategy Settings
INITIAL_CASH = 10000.0
PRODUCTION_MIN_CONFIDENCE = 0.65  # Buy if >65% confidence (2% in 5 days)
JACKPOT_MIN_CONFIDENCE = 0.55     # Buy if >55% confidence (20% in 20 days)
PRODUCTION_HOLD_PERIOD = 5
JACKPOT_HOLD_PERIOD = 20
SEQ_LEN = 60  # Need 60 days of history for prediction

os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.style.use('dark_background')

# --- LOAD MODELS ---

def load_model_components(model_path, features_path, scaler_path):
    """Load model, feature map, and scaler."""
    print(f"Loading model from {model_path.name}...")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load model with custom objects
    custom_objects = {
        'WarmupCosineDecay': WarmupCosineDecay
    }
    
    try:
        model = keras.models.load_model(model_path, custom_objects=custom_objects)
    except Exception as e:
        # If that fails, try loading without compiling (weights only)
        print(f"  ⚠️  Warning: Loading model without optimizer (weights only)")
        try:
            model = keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
            # Recompile with a simple optimizer (we don't need the exact LR schedule for inference)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        except Exception as e2:
            print(f"  ❌ Could not load model: {e2}")
            raise
    
    # Load feature map
    with open(features_path, 'r') as f:
        features = json.load(f)
    
    # Load scaler (optional - we'll use local scaling like in training)
    scaler = None
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
    
    print(f"  ✓ Model loaded: {len(features)} features")
    return model, features, scaler

# --- DATA LOADING ---

def load_stock_data(ticker, feature_cols):
    """Load processed stock data with proper feature alignment."""
    file_path = PROCESSED_DIR / f"{ticker}_daily_processed.csv"
    
    if not file_path.exists():
        return None
    
    try:
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        
        # Keep close for returns calculation
        if 'close' not in df.columns:
            return None
        
        # Align features (use reindex to handle missing columns)
        df_features = df.reindex(columns=feature_cols, fill_value=0)
        df_features = df_features.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Keep close and date for tracking
        df_features['close'] = df['close']
        
        return df_features
        
    except Exception as e:
        return None

def prepare_sequence(df, idx, feature_cols):
    """Prepare 60-day sequence for prediction."""
    if idx < SEQ_LEN:
        return None
    
    # Get 60 days of features
    seq_data = df.iloc[idx-SEQ_LEN:idx][feature_cols].values
    
    # Local scaling (like in training)
    scaler = StandardScaler()
    seq_scaled = scaler.fit_transform(seq_data).astype(np.float32)
    
    # Reshape for model: (1, 60, n_features)
    seq_reshaped = seq_scaled.reshape(1, SEQ_LEN, -1)
    
    return seq_reshaped

# --- SIMULATION ---

def run_backtest(model, features, mode="production"):
    """
    Run backtest on all stocks.
    
    Args:
        model: Loaded Keras model
        features: List of feature column names
        mode: "production" or "jackpot"
    """
    
    # Settings based on mode
    if mode == "production":
        min_conf = PRODUCTION_MIN_CONFIDENCE
        hold_period = PRODUCTION_HOLD_PERIOD
        target_gain = 0.02  # 2%
    else:
        min_conf = JACKPOT_MIN_CONFIDENCE
        hold_period = JACKPOT_HOLD_PERIOD
        target_gain = 0.20  # 20%
    
    print(f"\n{'='*60}")
    print(f"Running {mode.upper()} Backtest")
    print(f"  Min Confidence: {min_conf*100:.0f}%")
    print(f"  Hold Period: {hold_period} days")
    print(f"  Target Gain: {target_gain*100:.0f}%")
    print(f"{'='*60}\n")
    
    # Get all stock files
    stock_files = list(PROCESSED_DIR.glob("*_daily_processed.csv"))
    print(f"Found {len(stock_files)} processed stocks")
    
    # Load limited set for backtesting (top liquid stocks)
    tickers = [f.stem.split('_')[0] for f in stock_files[:500]]  # Top 500 stocks
    
    # Load stock data
    stock_data = {}
    for ticker in tickers:
        df = load_stock_data(ticker, features)
        if df is not None and len(df) > SEQ_LEN + hold_period:
            stock_data[ticker] = df
    
    print(f"Loaded {len(stock_data)} stocks with sufficient history\n")
    
    # Find common date range
    all_dates = set()
    for df in stock_data.values():
        all_dates.update(df.index)
    
    common_dates = sorted(list(all_dates))
    
    # Limit to last 2 years for faster backtesting
    cutoff_date = datetime.now() - timedelta(days=730)
    common_dates = [d for d in common_dates if d >= cutoff_date]
    
    print(f"Backtesting over {len(common_dates)} days ({common_dates[0].date()} to {common_dates[-1].date()})\n")
    
    # Run simulation
    cash = INITIAL_CASH
    portfolio_values = []
    trade_log = []
    current_position = None
    
    for i, date in enumerate(common_dates):
        
        # Close existing position
        if current_position and i >= current_position['exit_idx']:
            ticker = current_position['ticker']
            
            if ticker in stock_data and date in stock_data[ticker].index:
                exit_price = stock_data[ticker].loc[date, 'close']
                entry_price = current_position['entry_price']
                
                actual_return = (exit_price - entry_price) / entry_price
                cash *= (1 + actual_return)
                
                trade_log.append({
                    'date': date,
                    'action': 'SELL',
                    'ticker': ticker,
                    'price': exit_price,
                    'return': actual_return * 100,
                    'balance': cash
                })
            
            current_position = None
        
        # Look for new position
        if current_position is None:
            best_ticker = None
            best_prob = 0
            
            for ticker, df in stock_data.items():
                if date not in df.index:
                    continue
                
                idx = df.index.get_loc(date)
                
                # Need enough history + future
                if idx < SEQ_LEN or idx + hold_period >= len(df):
                    continue
                
                # Prepare sequence
                seq = prepare_sequence(df, idx, features)
                if seq is None:
                    continue
                
                # Predict
                try:
                    pred_prob = float(model.predict(seq, verbose=0)[0][0])
                    
                    if pred_prob > min_conf and pred_prob > best_prob:
                        best_prob = pred_prob
                        best_ticker = ticker
                except:
                    continue
            
            # Enter position
            if best_ticker:
                entry_price = stock_data[best_ticker].loc[date, 'close']
                
                current_position = {
                    'ticker': best_ticker,
                    'entry_idx': i,
                    'exit_idx': i + hold_period,
                    'entry_price': entry_price,
                    'confidence': best_prob
                }
                
                trade_log.append({
                    'date': date,
                    'action': 'BUY',
                    'ticker': best_ticker,
                    'price': entry_price,
                    'confidence': best_prob * 100,
                    'balance': cash
                })
        
        portfolio_values.append(cash)
    
    return common_dates, portfolio_values, trade_log

# --- PLOTTING ---

def plot_performance(dates, strat_values, trade_log, mode="production"):
    """Generate performance chart vs SPY."""
    
    if not dates or not strat_values:
        print("❌ No data to plot")
        return
    
    # Load SPY
    spy_values = []
    if SPY_PATH.exists():
        try:
            spy_df = pd.read_csv(SPY_PATH)
            spy_df['date'] = pd.to_datetime(spy_df['date'])
            spy_df.set_index('date', inplace=True)
            spy_aligned = spy_df.reindex(dates, method='ffill')
            
            start_price = spy_aligned['close'].iloc[0]
            spy_norm = (spy_aligned['close'] / start_price) * INITIAL_CASH
            spy_values = spy_norm.values
        except Exception as e:
            print(f"⚠️ SPY data error: {e}")
            spy_values = [INITIAL_CASH] * len(dates)
    else:
        spy_values = [INITIAL_CASH] * len(dates)
    
    # Plot
    fig, ax = plt.subplots(figsize=(16, 9))
    
    ax.plot(dates, spy_values, label="S&P 500 (SPY)", color="white", alpha=0.5, linewidth=2)
    ax.plot(dates, strat_values, label=f"AI Strategy ({mode.title()})", color="#00ff00", linewidth=3)
    
    # Fill between
    valid_len = min(len(dates), len(strat_values), len(spy_values))
    d_view = np.array(dates)[:valid_len]
    s_view = np.array(strat_values)[:valid_len]
    spy_view = np.array(spy_values)[:valid_len]
    
    mask_beat = (s_view > spy_view)
    mask_lose = (s_view <= spy_view)
    
    ax.fill_between(d_view, s_view, spy_view, where=mask_beat, interpolate=True, color='green', alpha=0.2)
    ax.fill_between(d_view, s_view, spy_view, where=mask_lose, interpolate=True, color='red', alpha=0.2)
    
    final_val = strat_values[-1]
    total_return = ((final_val / INITIAL_CASH) - 1) * 100
    
    ax.set_title(f"{mode.title()} Model - Final: ${final_val:,.2f} ({total_return:+.1f}%)", 
                 fontsize=20, color="white")
    ax.set_ylabel("Portfolio Value ($)", color="white")
    ax.set_xlabel("Date", color="white")
    ax.grid(True, alpha=0.1)
    ax.legend()
    
    out_path = OUTPUT_DIR / f"{mode}_backtest_performance.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📸 Chart saved: {out_path}")

# --- MAIN ---

def main():
    print("\n" + "="*60)
    print("LSTM MODEL BACKTESTING SYSTEM")
    print("="*60 + "\n")
    
    # Choose which model to backtest
    print("Select model to backtest:")
    print("  1. Production Model (2% in 5 days)")
    print("  2. Jackpot Model (20% in 20 days)")
    print("  3. Both")
    
    choice = input("\nChoice (1/2/3): ").strip()
    
    models_to_test = []
    
    if choice in ['1', '3']:
        try:
            prod_model, prod_features, prod_scaler = load_model_components(
                PRODUCTION_MODEL, PRODUCTION_FEATURES, PRODUCTION_SCALER
            )
            models_to_test.append(('production', prod_model, prod_features))
        except Exception as e:
            print(f"❌ Failed to load production model: {e}")
    
    if choice in ['2', '3']:
        try:
            jack_model, jack_features, jack_scaler = load_model_components(
                JACKPOT_MODEL, JACKPOT_FEATURES, JACKPOT_SCALER
            )
            models_to_test.append(('jackpot', jack_model, jack_features))
        except Exception as e:
            print(f"❌ Failed to load jackpot model: {e}")
    
    if not models_to_test:
        print("❌ No models loaded. Exiting.")
        return
    
    # Run backtests
    for mode, model, features in models_to_test:
        dates, values, log = run_backtest(model, features, mode)
        
        if not values:
            print(f"❌ {mode} backtest failed")
            continue
        
        # Save trade log
        df_log = pd.DataFrame(log)
        log_path = OUTPUT_DIR / f"{mode}_trade_log.csv"
        df_log.to_csv(log_path, index=False)
        
        # Calculate stats
        sells = df_log[df_log['action'] == 'SELL']
        if len(sells) > 0:
            wins = sells[sells['return'] > 0]
            win_rate = len(wins) / len(sells)
            avg_win = wins['return'].mean() if len(wins) > 0 else 0
            avg_loss = sells[sells['return'] <= 0]['return'].mean() if len(sells[sells['return'] <= 0]) > 0 else 0
            
            print(f"\n{'='*60}")
            print(f"{mode.upper()} RESULTS")
            print(f"{'='*60}")
            print(f"💰 Final Balance:   ${values[-1]:,.2f}")
            print(f"📈 Total Return:    {((values[-1]/INITIAL_CASH)-1)*100:+.2f}%")
            print(f"🎯 Win Rate:        {win_rate:.1%}")
            print(f"📊 Total Trades:    {len(sells)}")
            print(f"✅ Avg Win:         {avg_win:+.2f}%")
            print(f"❌ Avg Loss:        {avg_loss:.2f}%")
            print(f"📁 Trade Log:       {log_path}")
            print(f"{'='*60}\n")
        
        # Plot
        plot_performance(dates, values, log, mode)
    
    print("\n✅ Backtesting complete!")
    print(f"📂 Results saved to: {OUTPUT_DIR}\n")

if __name__ == "__main__":
    main()