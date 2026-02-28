"""
LSTM Stock Predictor — Backtest Engine v2

Phase 2 improvements over v1:
  - Multi-position portfolio (N_POSITIONS=10 equal-weight slots)
  - Transaction costs + tiered slippage (0.10 / 0.20 / 0.50% by avg daily volume)
  - Liquidity filter: price > $5, avg daily volume > 100k shares/day
  - SPY buy-and-hold benchmark (fetched via yfinance, cached locally)
  - Full risk metrics: Sharpe, Sortino, max drawdown, Calmar, alpha, beta
  - Per-year performance breakdown (walk-forward diagnostic)
"""

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    print("Warning: yfinance not available — SPY benchmark will be skipped.")


# ---------------------------------------------------------------------------
# Custom LR schedule (must be registered before loading saved models)
# ---------------------------------------------------------------------------

@keras.utils.register_keras_serializable(package="Custom", name="WarmupCosineDecay")
class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, warmup_steps, total_decay_steps=None, initial_lr=1e-3,
                 warmup_start_lr=1e-5, t_mul=2.0, m_mul=0.9, **kwargs):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        self.warmup_start_lr = warmup_start_lr
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
        ws = tf.cast(self.warmup_steps, tf.float32)
        warmup_lr = self.warmup_start_lr + (self.initial_lr - self.warmup_start_lr) * (step / ws)
        cosine_lr = self.cosine_schedule(step - ws)
        return tf.where(step < ws, warmup_lr, cosine_lr)

    def get_config(self):
        return {"warmup_steps": self.warmup_steps,
                "initial_lr": self.initial_lr,
                "warmup_start_lr": self.warmup_start_lr}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROCESSED_DIR = Path("TrainingData/indicators_data/processed/stocksData")
OUTPUT_DIR    = Path("backtest_results")
OUTPUT_DIR.mkdir(exist_ok=True)

INITIAL_CASH  = 10_000.0
N_POSITIONS   = 10          # max simultaneous open positions
UNIVERSE_SIZE = 500         # stocks scanned each backtest run
BACKTEST_DAYS = 730         # ~2 calendar years of history
SEQ_LEN       = 60
RISK_FREE_RATE = 0.05       # annualised (5% — appropriate for 2024-2026)

# Thresholds match daily_picks.py so backtest and live inference are comparable
PRODUCTION_MIN_CONFIDENCE = 0.60
JACKPOT_MIN_CONFIDENCE    = 0.55
PRODUCTION_HOLD_PERIOD    = 5
JACKPOT_HOLD_PERIOD       = 20

# Liquidity filters — applied at position-entry time
MIN_PRICE   = 5.0           # $/share
MIN_AVG_VOL = 100_000       # shares/day (20-day average)

# Slippage tiers: (min_avg_vol_threshold, one-way rate)
# Applied on both entry and exit, so round-trip = 2× the rate shown here.
SLIPPAGE_TIERS = [
    (1_000_000, 0.0010),   # > 1M shares/day  →  0.10% per side
    (100_000,   0.0020),   # 100k–1M           →  0.20% per side
    (0,         0.0050),   # < 100k (post-filter edge cases) → 0.50%
]


# ---------------------------------------------------------------------------
# Utility — slippage
# ---------------------------------------------------------------------------

def get_slippage_rate(vol_sma_20: float) -> float:
    for threshold, rate in SLIPPAGE_TIERS:
        if vol_sma_20 >= threshold:
            return rate
    return 0.005


# ---------------------------------------------------------------------------
# Utility — risk metrics
# ---------------------------------------------------------------------------

def _rf_daily():
    return (1 + RISK_FREE_RATE) ** (1 / 252) - 1


def compute_sharpe(daily_returns: pd.Series) -> float:
    rf = _rf_daily()
    excess = daily_returns - rf
    return float(np.sqrt(252) * excess.mean() / excess.std()) if excess.std() > 1e-10 else 0.0


def compute_sortino(daily_returns: pd.Series) -> float:
    rf = _rf_daily()
    excess = daily_returns - rf
    downside_std = daily_returns[daily_returns < rf].std()
    return float(np.sqrt(252) * excess.mean() / downside_std) if downside_std > 1e-10 else 0.0


def compute_max_drawdown(portfolio_values) -> float:
    vals = np.asarray(portfolio_values, dtype=float)
    peak = np.maximum.accumulate(vals)
    dd = (vals - peak) / np.where(peak > 0, peak, 1)
    return float(dd.min())   # negative fraction, e.g. -0.23 = -23%


def compute_cagr(portfolio_values, n_trading_days: int) -> float:
    if n_trading_days < 1:
        return 0.0
    total = portfolio_values[-1] / portfolio_values[0]
    years = n_trading_days / 252
    return float(total ** (1 / years) - 1)


def compute_calmar(cagr: float, max_dd: float) -> float:
    return cagr / abs(max_dd) if abs(max_dd) > 1e-10 else 0.0


def compute_beta_alpha(port_returns: np.ndarray, spy_returns: np.ndarray) -> tuple:
    """Returns (beta, annualised_alpha_fraction)."""
    valid = np.isfinite(port_returns) & np.isfinite(spy_returns)
    p, s = port_returns[valid], spy_returns[valid]
    if len(p) < 30:
        return 0.0, 0.0
    cov = np.cov(p, s)
    beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 1e-10 else 0.0
    rf = _rf_daily()
    alpha_daily = p.mean() - rf - beta * (s.mean() - rf)
    return float(beta), float(alpha_daily * 252)


# ---------------------------------------------------------------------------
# SPY benchmark
# ---------------------------------------------------------------------------

def fetch_spy_data(start: datetime, end: datetime) -> pd.Series | None:
    """
    Returns a pd.Series of SPY daily close prices indexed by date.
    Tries a local cache first, then yfinance.
    """
    cache_path = OUTPUT_DIR / "spy_cache.csv"

    # Try cache
    if cache_path.exists():
        try:
            spy = pd.read_csv(cache_path, index_col=0, parse_dates=True)['close']
            spy.index = pd.DatetimeIndex(spy.index).tz_localize(None)
            cached_start = spy.index.min()
            cached_end   = spy.index.max()
            if cached_start <= pd.Timestamp(start) and cached_end >= pd.Timestamp(end):
                return spy
        except Exception:
            pass

    if not HAS_YFINANCE:
        return None

    print("  Fetching SPY data from Yahoo Finance...")
    # yf.download() has an internal TypeError bug in several recent releases.
    # Ticker.history() uses a different code path and is more stable.
    try:
        ticker = yf.Ticker('SPY')
        raw = ticker.history(
            start=(start - timedelta(days=5)).strftime('%Y-%m-%d'),
            end=(end + timedelta(days=5)).strftime('%Y-%m-%d'),
            auto_adjust=True,
        )
        if raw is None or raw.empty:
            print("  SPY fetch returned empty data.")
            return None
        raw.index = pd.DatetimeIndex(raw.index).tz_localize(None)
        spy = raw['Close'].rename('close')
        pd.DataFrame({'close': spy}).to_csv(cache_path)
        print(f"  SPY cached to {cache_path}")
        return spy
    except Exception as e:
        print(f"  SPY fetch failed: {e}")
        return None


def build_spy_equity_curve(spy_prices: pd.Series, dates: list) -> np.ndarray | None:
    """Buy-and-hold SPY equity curve aligned to our backtest dates."""
    if spy_prices is None:
        return None
    dt_index = pd.DatetimeIndex(dates).tz_localize(None)
    spy_aligned = spy_prices.reindex(dt_index, method='ffill').dropna()
    if spy_aligned.empty:
        return None
    start_price = spy_aligned.iloc[0]
    curve = (spy_aligned / start_price * INITIAL_CASH).values
    # Pad/trim to match len(dates)
    result = np.full(len(dates), np.nan)
    for i, d in enumerate(dt_index):
        if d in spy_aligned.index:
            result[i] = spy_aligned[d] / start_price * INITIAL_CASH
    # Forward-fill gaps (weekends already excluded from dates, but just in case)
    mask = np.isfinite(result)
    if mask.any():
        result = pd.Series(result).ffill().bfill().values
    return result


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_components(model_path: Path, features_path: Path, scaler_path: Path):
    print(f"Loading model: {model_path.name} ...")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    custom_objects = {'WarmupCosineDecay': WarmupCosineDecay}
    try:
        model = keras.models.load_model(model_path, custom_objects=custom_objects)
    except Exception:
        model = keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    with open(features_path) as f:
        features = json.load(f)

    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    print(f"  ✓ {len(features)} features")
    return model, features, scaler


# ---------------------------------------------------------------------------
# Data loading + liquidity filter
# ---------------------------------------------------------------------------

def load_stock_data(ticker: str, feature_cols: list) -> pd.DataFrame | None:
    """
    Load processed stock data.

    Returns a DataFrame with columns = feature_cols + ['close', 'vol_sma_20'],
    indexed by date.  Returns None if the stock fails the liquidity filter.
    """
    file_path = PROCESSED_DIR / f"{ticker}_daily_processed.csv"
    if not file_path.exists():
        return None

    try:
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)

        if 'close' not in df.columns:
            return None

        close_series   = pd.to_numeric(df['close'], errors='coerce').fillna(0)
        vol_sma_series = pd.to_numeric(df.get('vol_sma_20', pd.Series(0, index=df.index)),
                                       errors='coerce').fillna(0)

        # Build feature matrix
        df_features = df.reindex(columns=feature_cols).copy()
        for col in df_features.columns:
            df_features[col] = pd.to_numeric(df_features[col], errors='coerce')
        df_features = df_features.replace([np.inf, -np.inf], np.nan).fillna(0)

        df_features['close']     = close_series
        df_features['vol_sma_20'] = vol_sma_series

        # --- Liquidity filter (applied on recent 60-bar median) ---
        recent = df_features.iloc[-min(60, len(df_features)):]
        med_price = recent['close'].median()
        med_vol   = recent['vol_sma_20'].median()
        if med_price < MIN_PRICE or med_vol < MIN_AVG_VOL:
            return None

        return df_features

    except Exception:
        return None


def prepare_sequence(df: pd.DataFrame, idx: int, feature_cols: list) -> np.ndarray | None:
    """Scale a 60-day window and return shape (1, SEQ_LEN, n_features)."""
    if idx < SEQ_LEN:
        return None
    try:
        seq_data = df.iloc[idx - SEQ_LEN:idx][feature_cols].values.astype(np.float64)
        if seq_data.shape != (SEQ_LEN, len(feature_cols)):
            return None
        if not np.isfinite(seq_data).all():
            seq_data = np.nan_to_num(seq_data, nan=0.0, posinf=0.0, neginf=0.0)
        scaled = StandardScaler().fit_transform(seq_data).astype(np.float32)
        return scaled.reshape(1, SEQ_LEN, len(feature_cols))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Backtest simulation
# ---------------------------------------------------------------------------

def run_backtest(model, features: list, mode: str = "production"):
    if mode == "production":
        min_conf    = PRODUCTION_MIN_CONFIDENCE
        hold_period = PRODUCTION_HOLD_PERIOD
    else:
        min_conf    = JACKPOT_MIN_CONFIDENCE
        hold_period = JACKPOT_HOLD_PERIOD

    print(f"\n{'='*65}")
    print(f"  {mode.upper()} BACKTEST")
    print(f"  Confidence threshold : {min_conf:.0%}")
    print(f"  Hold period          : {hold_period} trading days")
    print(f"  Portfolio slots      : {N_POSITIONS}")
    print(f"  Liquidity filter     : price > ${MIN_PRICE}, vol > {MIN_AVG_VOL:,}/day")
    print(f"  Slippage             : tiered (0.10 / 0.20 / 0.50% per side)")
    print(f"{'='*65}\n")

    # --- Load universe ---
    all_files = sorted(PROCESSED_DIR.glob("*_daily_processed.csv"))[:UNIVERSE_SIZE]
    tickers   = [f.stem.split('_')[0] for f in all_files]

    stock_data: dict[str, pd.DataFrame] = {}
    n_filtered = 0
    print(f"Loading up to {UNIVERSE_SIZE} stocks (applying liquidity filter)...")
    for ticker in tickers:
        df = load_stock_data(ticker, features)
        if df is None:
            n_filtered += 1
            continue
        if len(df) > SEQ_LEN + hold_period:
            stock_data[ticker] = df

    n_passed = len(stock_data)
    print(f"  Passed liquidity filter : {n_passed}")
    print(f"  Filtered out            : {n_filtered}")

    if n_passed == 0:
        print("No stocks survived the filter. Exiting.")
        return [], [], [], []

    # --- Date range (last BACKTEST_DAYS calendar days) ---
    all_dates: set = set()
    for df in stock_data.values():
        all_dates.update(df.index)
    all_dates = sorted(all_dates)

    cutoff = datetime.now() - timedelta(days=BACKTEST_DAYS)
    common_dates = [d for d in all_dates
                    if pd.Timestamp(cutoff) <= d <= pd.Timestamp(datetime.now())]
    if not common_dates:
        print("No common dates in backtest window.")
        return [], [], [], []

    print(f"  Date range : {common_dates[0].date()} → {common_dates[-1].date()}")
    print(f"  Trading days: {len(common_dates)}\n")

    # --- Portfolio state ---
    cash      = INITIAL_CASH
    positions = {}   # ticker → {size, entry_price, exit_date, confidence}
    portfolio_values = []
    trade_log        = []
    n_predictions    = 0
    n_above_threshold = 0

    for i, date in enumerate(common_dates):

        # === STEP 1: Close expired positions ===
        tickers_to_close = [t for t, p in positions.items()
                            if p['exit_date'] is not None and p['exit_date'] <= date]
        for ticker in tickers_to_close:
            pos = positions.pop(ticker)
            df  = stock_data[ticker]
            # Use closest available price on or before exit date
            avail = df.index[df.index <= date]
            exit_price = float(df.loc[avail[-1], 'close']) if len(avail) > 0 else pos['entry_price']
            vol_sma    = float(df.loc[avail[-1], 'vol_sma_20']) if len(avail) > 0 else MIN_AVG_VOL
            slip       = get_slippage_rate(vol_sma)
            proceeds   = pos['size'] * (exit_price / pos['entry_price']) * (1 - slip)
            cash      += proceeds
            trade_log.append({
                'date'        : date,
                'action'      : 'SELL',
                'ticker'      : ticker,
                'price'       : round(exit_price, 4),
                'entry_price' : round(pos['entry_price'], 4),
                'return_pct'  : round((proceeds / pos['size'] - 1) * 100, 3),
                'slippage_pct': round(slip * 100, 2),
                'confidence'  : round(pos['confidence'] * 100, 2),
            })

        # === STEP 2: Mark-to-market portfolio value ===
        pos_value = 0.0
        for ticker, pos in positions.items():
            df = stock_data[ticker]
            avail = df.index[df.index <= date]
            price = float(df.loc[avail[-1], 'close']) if len(avail) > 0 else pos['entry_price']
            pos_value += pos['size'] * (price / pos['entry_price'])
        portfolio_value = cash + pos_value
        portfolio_values.append(portfolio_value)

        # === STEP 3: Open new positions when slots are available ===
        n_available = N_POSITIONS - len(positions)
        if n_available == 0:
            continue

        # Scan universe for predictions above threshold
        candidates = []
        for ticker, df in stock_data.items():
            if ticker in positions:
                continue
            if date not in df.index:
                continue
            idx = df.index.get_loc(date)
            if idx < SEQ_LEN or idx + hold_period >= len(df):
                continue

            # Per-entry liquidity check (prices/volumes change over time)
            entry_price = float(df.loc[date, 'close'])
            vol_sma     = float(df.loc[date, 'vol_sma_20'])
            if entry_price < MIN_PRICE or vol_sma < MIN_AVG_VOL:
                continue

            seq = prepare_sequence(df, idx, features)
            if seq is None:
                continue
            try:
                prob = float(model.predict(seq, verbose=0)[0][0])
                n_predictions += 1
                if prob >= min_conf:
                    n_above_threshold += 1
                    candidates.append((ticker, prob, entry_price, vol_sma))
            except Exception:
                continue

        # Rank by confidence, fill available slots
        candidates.sort(key=lambda x: x[1], reverse=True)
        slot_size = portfolio_value / N_POSITIONS  # equal-weight target per slot

        for ticker, prob, entry_price, vol_sma in candidates[:n_available]:
            if cash < slot_size * 0.10:   # need at least 10% of a slot to open
                break
            slip         = get_slippage_rate(vol_sma)
            actual_size  = min(slot_size, cash / (1 + slip))
            cost         = actual_size * (1 + slip)
            if cost > cash:
                continue
            # Determine exit date using trading-day index
            exit_idx  = min(i + hold_period, len(common_dates) - 1)
            exit_date = common_dates[exit_idx]
            cash -= cost
            positions[ticker] = {
                'size'        : actual_size,
                'entry_price' : entry_price,
                'exit_date'   : exit_date,
                'confidence'  : prob,
            }
            trade_log.append({
                'date'        : date,
                'action'      : 'BUY',
                'ticker'      : ticker,
                'price'       : round(entry_price, 4),
                'confidence'  : round(prob * 100, 2),
                'size'        : round(actual_size, 2),
                'slippage_pct': round(slip * 100, 2),
            })

    # Close any positions still open at end of backtest
    final_date = common_dates[-1]
    for ticker, pos in list(positions.items()):
        df    = stock_data[ticker]
        avail = df.index[df.index <= final_date]
        exit_price = float(df.loc[avail[-1], 'close']) if len(avail) > 0 else pos['entry_price']
        vol_sma    = float(df.loc[avail[-1], 'vol_sma_20']) if len(avail) > 0 else MIN_AVG_VOL
        slip       = get_slippage_rate(vol_sma)
        proceeds   = pos['size'] * (exit_price / pos['entry_price']) * (1 - slip)
        cash      += proceeds
        trade_log.append({
            'date'        : final_date,
            'action'      : 'SELL',
            'ticker'      : ticker,
            'price'       : round(exit_price, 4),
            'entry_price' : round(pos['entry_price'], 4),
            'return_pct'  : round((proceeds / pos['size'] - 1) * 100, 3),
            'slippage_pct': round(slip * 100, 2),
            'confidence'  : round(pos['confidence'] * 100, 2),
            'note'        : 'forced close at end of backtest',
        })
    positions.clear()
    portfolio_values[-1] = cash   # update last value after forced closes

    print(f"Prediction statistics:")
    print(f"  Total predictions made  : {n_predictions:,}")
    print(f"  Above threshold ({min_conf:.0%})   : {n_above_threshold:,}")
    if n_predictions > 0:
        print(f"  Pass rate               : {n_above_threshold / n_predictions:.1%}")

    daily_returns = pd.Series(portfolio_values).pct_change().dropna()
    return common_dates, portfolio_values, trade_log, daily_returns


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_all_metrics(portfolio_values, daily_returns, dates, spy_prices, trade_log):
    sells = [t for t in trade_log if t['action'] == 'SELL' and 'return_pct' in t]

    cagr   = compute_cagr(portfolio_values, len(dates))
    max_dd = compute_max_drawdown(portfolio_values)
    sharpe = compute_sharpe(daily_returns)
    sortino = compute_sortino(daily_returns)
    calmar  = compute_calmar(cagr, max_dd)

    win_returns  = [t['return_pct'] for t in sells if t['return_pct'] > 0]
    loss_returns = [t['return_pct'] for t in sells if t['return_pct'] <= 0]
    win_rate = len(win_returns) / len(sells) if sells else 0.0
    avg_win  = float(np.mean(win_returns))  if win_returns  else 0.0
    avg_loss = float(np.mean(loss_returns)) if loss_returns else 0.0
    profit_factor = (sum(win_returns) / abs(sum(loss_returns))
                     if loss_returns and sum(loss_returns) != 0 else float('inf'))

    # SPY alignment
    beta, alpha, spy_return = 0.0, 0.0, None
    if spy_prices is not None:
        dt_index = pd.DatetimeIndex(dates).tz_localize(None)
        spy_aligned = spy_prices.reindex(dt_index, method='ffill').dropna()
        if len(spy_aligned) >= 30:
            spy_rets = spy_aligned.pct_change().dropna().values
            port_rets_np = daily_returns.values[:len(spy_rets)]
            beta, alpha = compute_beta_alpha(port_rets_np, spy_rets)
            spy_return = float(spy_aligned.iloc[-1] / spy_aligned.iloc[0] - 1)

    return {
        'total_return' : portfolio_values[-1] / INITIAL_CASH - 1,
        'cagr'         : cagr,
        'sharpe'       : sharpe,
        'sortino'      : sortino,
        'max_drawdown' : max_dd,
        'calmar'       : calmar,
        'beta'         : beta,
        'alpha'        : alpha,
        'spy_return'   : spy_return,
        'win_rate'     : win_rate,
        'n_trades'     : len(sells),
        'avg_win_pct'  : avg_win,
        'avg_loss_pct' : avg_loss,
        'profit_factor': profit_factor,
        'final_value'  : portfolio_values[-1],
    }


def print_results(metrics: dict, mode: str):
    spy_line = (f"  SPY buy-and-hold       : {metrics['spy_return']:+.1%}"
                if metrics['spy_return'] is not None else "  SPY buy-and-hold       : N/A")
    alpha_pct = metrics['alpha'] * 100 if abs(metrics['alpha']) < 10 else metrics['alpha']

    print(f"\n{'='*65}")
    print(f"  {mode.upper()} RESULTS")
    print(f"{'='*65}")
    print(f"  Final portfolio value  : ${metrics['final_value']:>10,.2f}")
    print(f"  Total return           : {metrics['total_return']:>+10.1%}")
    print(f"  CAGR                   : {metrics['cagr']:>+10.1%}")
    print(spy_line)
    print(f"{'—'*65}")
    print(f"  Sharpe ratio           : {metrics['sharpe']:>10.3f}")
    print(f"  Sortino ratio          : {metrics['sortino']:>10.3f}")
    print(f"  Max drawdown           : {metrics['max_drawdown']:>+10.1%}")
    print(f"  Calmar ratio           : {metrics['calmar']:>10.3f}")
    print(f"  Beta (vs SPY)          : {metrics['beta']:>10.3f}")
    print(f"  Alpha (annualised)     : {alpha_pct:>+10.2f}%")
    print(f"{'—'*65}")
    print(f"  Total closed trades    : {metrics['n_trades']:>10}")
    print(f"  Win rate               : {metrics['win_rate']:>10.1%}")
    print(f"  Avg win                : {metrics['avg_win_pct']:>+10.2f}%")
    print(f"  Avg loss               : {metrics['avg_loss_pct']:>+10.2f}%")
    print(f"  Profit factor          : {metrics['profit_factor']:>10.2f}x")
    print(f"{'='*65}\n")


def print_yearly_breakdown(dates, portfolio_values, spy_equity, trade_log, mode):
    """Per-year return, Sharpe, max drawdown, win rate, SPY return."""
    dt_index = pd.DatetimeIndex(dates)
    years    = sorted(dt_index.year.unique())

    rows = []
    for year in years:
        mask = [d.year == year for d in dt_index]
        y_vals = [v for v, m in zip(portfolio_values, mask) if m]
        if len(y_vals) < 2:
            continue

        y_return = y_vals[-1] / y_vals[0] - 1
        y_rets   = pd.Series(y_vals).pct_change().dropna()
        y_sharpe = compute_sharpe(y_rets)
        y_mdd    = compute_max_drawdown(y_vals)
        y_trades = [t for t in trade_log
                    if t['action'] == 'SELL' and 'return_pct' in t
                    and pd.Timestamp(t['date']).year == year]
        y_wr     = (sum(1 for t in y_trades if t['return_pct'] > 0) / len(y_trades)
                    if y_trades else float('nan'))

        spy_yr = None
        if spy_equity is not None:
            y_spy = [v for v, m in zip(spy_equity, mask) if m and np.isfinite(v)]
            if len(y_spy) >= 2:
                spy_yr = y_spy[-1] / y_spy[0] - 1

        rows.append({
            'Year'      : year,
            'Return'    : f"{y_return:+.1%}",
            'Sharpe'    : f"{y_sharpe:.2f}",
            'Max DD'    : f"{y_mdd:.1%}",
            'Win Rate'  : f"{y_wr:.0%}" if np.isfinite(y_wr) else "—",
            'Trades'    : len(y_trades),
            'SPY Return': f"{spy_yr:+.1%}" if spy_yr is not None else "N/A",
        })

    if rows:
        df_yr = pd.DataFrame(rows)
        print(f"\n{'='*65}")
        print(f"  {mode.upper()} — YEAR-BY-YEAR BREAKDOWN")
        print(f"{'='*65}")
        print(df_yr.to_string(index=False))
        print(f"{'='*65}\n")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_performance(dates, portfolio_values, spy_equity, trade_log, mode: str):
    """
    3-panel chart:
      Top    — equity curve (strategy vs SPY)
      Middle — drawdown (strategy)
      Bottom — rolling 30-trade win rate
    """
    dates_dt = pd.DatetimeIndex(dates)
    vals     = np.array(portfolio_values)
    peak     = np.maximum.accumulate(vals)
    drawdown = (vals - peak) / peak * 100  # %

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(3, 1, height_ratios=[3, 1.5, 1.5], hspace=0.35)

    # --- Equity curve ---
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(dates_dt, vals, linewidth=1.8, color='#2196F3', label='Strategy', zorder=3)
    ax1.axhline(INITIAL_CASH, color='grey', linestyle='--', linewidth=0.8, label='Initial capital')
    if spy_equity is not None:
        ax1.plot(dates_dt, spy_equity, linewidth=1.4, color='#FF9800',
                 linestyle='--', label='SPY buy-and-hold', alpha=0.85)
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.set_title(f'{mode.upper()} Model Backtest  —  {N_POSITIONS}-slot equal-weight portfolio',
                  fontsize=12, fontweight='bold')
    ax1.legend(framealpha=0.9)
    ax1.grid(alpha=0.25)

    # --- Drawdown ---
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.fill_between(dates_dt, drawdown, 0, color='#F44336', alpha=0.55, label='Strategy drawdown')
    if spy_equity is not None:
        spy_peak = np.maximum.accumulate(spy_equity)
        spy_dd   = (spy_equity - spy_peak) / spy_peak * 100
        ax2.plot(dates_dt, spy_dd, color='#FF9800', linewidth=0.9,
                 linestyle='--', alpha=0.75, label='SPY drawdown')
    ax2.set_ylabel('Drawdown (%)')
    ax2.legend(fontsize=8, framealpha=0.9)
    ax2.grid(alpha=0.25)

    # --- Rolling 30-trade win rate ---
    ax3 = fig.add_subplot(gs[2])
    sells = [(pd.Timestamp(t['date']), t['return_pct'])
             for t in trade_log if t['action'] == 'SELL' and 'return_pct' in t]
    if len(sells) >= 5:
        sell_dates, sell_rets = zip(*sells)
        sell_dates = pd.DatetimeIndex(sell_dates)
        sell_rets  = pd.Series(sell_rets, index=sell_dates)
        window     = min(30, len(sell_rets))
        rolling_wr = (sell_rets > 0).rolling(window).mean() * 100
        ax3.plot(sell_dates, rolling_wr, color='#4CAF50', linewidth=1.4,
                 label=f'Rolling {window}-trade win rate')
        ax3.axhline(50, color='grey', linestyle='--', linewidth=0.7)
        ax3.set_ylabel('Win rate (%)')
        ax3.legend(fontsize=8, framealpha=0.9)
        ax3.grid(alpha=0.25)
    else:
        ax3.text(0.5, 0.5, 'Insufficient trades for rolling win rate',
                 ha='center', va='center', transform=ax3.transAxes, color='grey')

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)

    chart_path = OUTPUT_DIR / f"{mode}_backtest_performance.png"
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Chart saved: {chart_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("  LSTM STOCK PREDICTOR — BACKTEST ENGINE v2")
    print("=" * 65)

    print("\nSelect model:")
    print("  1. Production  (>2% in 5 days)")
    print("  2. Jackpot     (>20% in 20 days)")
    print("  3. Both")
    choice = input("Choice (1/2/3): ").strip()

    models_to_test = []

    if choice in ('1', '3'):
        try:
            m, f, s = load_model_components(
                Path("TrainingData/models/lstm_production.keras"),
                Path("TrainingData/models/feature_columns_production.json"),
                Path("TrainingData/models/scaler_production.joblib"),
            )
            models_to_test.append(("production", m, f))
        except Exception as e:
            print(f"  Could not load production model: {e}")

    if choice in ('2', '3'):
        try:
            m, f, s = load_model_components(
                Path("TrainingData/models/lstm_jackpot.keras"),
                Path("TrainingData/models/feature_columns_jackpot.json"),
                Path("TrainingData/models/scaler_jackpot.joblib"),
            )
            models_to_test.append(("jackpot", m, f))
        except Exception as e:
            print(f"  Could not load jackpot model: {e}")

    if not models_to_test:
        print("No models loaded. Exiting.")
        return

    for mode, model, features in models_to_test:
        dates, portfolio_values, trade_log, daily_returns = run_backtest(model, features, mode)

        if not portfolio_values:
            print(f"  {mode}: backtest returned no data.")
            continue

        # Fetch SPY benchmark
        spy_prices = fetch_spy_data(
            start=dates[0].to_pydatetime(),
            end=dates[-1].to_pydatetime(),
        )
        spy_equity = build_spy_equity_curve(spy_prices, dates)

        # Metrics
        metrics = compute_all_metrics(
            portfolio_values, daily_returns, dates, spy_prices, trade_log
        )
        print_results(metrics, mode)
        print_yearly_breakdown(dates, portfolio_values, spy_equity, trade_log, mode)

        # Save outputs
        df_log = pd.DataFrame(trade_log)
        log_path = OUTPUT_DIR / f"{mode}_trade_log.csv"
        df_log.to_csv(log_path, index=False)
        print(f"  Trade log saved : {log_path}")

        metrics_path = OUTPUT_DIR / f"{mode}_metrics.json"
        with open(metrics_path, 'w') as f_out:
            json.dump({k: (round(v, 6) if isinstance(v, float) else v)
                       for k, v in metrics.items()}, f_out, indent=2)
        print(f"  Metrics saved   : {metrics_path}")

        plot_performance(dates, portfolio_values, spy_equity, trade_log, mode)

    print("\nBacktest complete.")
    print(f"Results in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
