import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
from pathlib import Path
plt.rcParams['image.cmap'] = 'viridis'
plt.rcParams['savefig.transparent'] = False
plt.rcParams['savefig.facecolor'] = 'black'
plt.rcParams['savefig.edgecolor'] = 'black'
from PIL import Image

def force_png_rgb(path):
    img = Image.open(path).convert("RGB")
    img.save(path)

PROJECT_ROOT = Path(os.getcwd())
video_dir    = PROJECT_ROOT / "videos"
forecast_dir = PROJECT_ROOT / "forecasts"
os.makedirs(video_dir, exist_ok=True)

# Config
initial_value   = 1.0
random_runs     = 10
SPIKE_THRESHOLD = 0.30      # Reject stocks with daily abs change > 30%
MIN_ACCEPTED    = 0.0       # Minimum adjusted probability required
STD_FACTOR      = 0.0       # AdjustedProb = PredProb - STD_FACTOR * StdDev

# Backtest horizons
PERIODS = {
    "1d": 1,
    "1w": 5,
    "1m": 21,
    "6m": 126,
}

all_forecasts = {}
rejected_tickers = []

for filename in os.listdir(forecast_dir):

    if not filename.endswith("_forecast.csv"):
        continue

    filepath = forecast_dir / filename
    ticker   = filename.split("_forecast")[0]

    df = pd.read_csv(filepath, parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    df["Ticker"] = ticker

    # Spike filter (I need to build something better for stock splits in the future lol)
    if "Close" in df.columns:
        df["pct_change"] = df["Close"].pct_change()
        max_change       = df["pct_change"].abs().max()

        if max_change > SPIKE_THRESHOLD:
            rejected_tickers.append((ticker, max_change))
            continue

        # Compute realized log returns for horizons
        for label, days in PERIODS.items():
            df[f"Actual_LogR_{label}"] = np.log(df["Close"].shift(-days) / df["Close"])

    all_forecasts[ticker] = df


# Report rejected tickers
if rejected_tickers:
    print("\nRejected tickers due to unrealistic daily spikes (>30%):")
    for t, m in rejected_tickers:
        print(f"  - {t}: {m:.2%}")
else:
    print("\nNo stocks rejected for excessive daily spikes.")


# Get common dates across all tickers

all_dates     = [set(df.index) for df in all_forecasts.values()]
common_dates  = sorted(set.intersection(*all_dates))

print(f"\nUsing {len(common_dates)} shared dates across {len(all_forecasts)} tickers")


# Backtesting strategy:
# Rules:
#   For each date, compute adj_prob = Pred_Prob - STD_FACTOR * Pred_Prob_Std
#   Reject candidates with adj_prob <= MIN_ACCEPTED
#   Out of all candidates, choose the one with the highest adj_prob
#   Only hold 1 position at a time

strategy_value  = initial_value
strategy_history = []
trade_log        = []
buy_points       = []

current_hold = None
successful_buys = 0
total_buys      = 0

for i, date in enumerate(common_dates):
    # Exit if horizon has expired
    if current_hold is not None and i == current_hold["exit_idx"]:

        realized_logr = current_hold["Actual_LogR"]
        realized_pct  = np.exp(realized_logr) - 1 if not np.isnan(realized_logr) else np.nan

        trade_log.append({
            "BuyDate"        : current_hold["BuyDate"],
            "SellDate"       : date,
            "Ticker"         : current_hold["Ticker"],
            "Horizon"        : current_hold["Period"],
            "DaysHeld"       : current_hold["Days"],
            "Pred_Prob"      : current_hold["Pred_Prob"],
            "Pred_Prob_Std"  : current_hold["Pred_Prob_Std"],
            "Adj_Prob"       : current_hold["Adj_Prob"],
            "Actual_LogR"    : realized_logr,
            "Actual_Return%" : realized_pct * 100 if not np.isnan(realized_pct) else np.nan,
        })

        if not np.isnan(realized_logr):
            strategy_value *= np.exp(realized_logr)
            if realized_logr > 0:
                successful_buys += 1

        current_hold = None

    # If still holding a position, skip new buys
    if current_hold is not None:
        strategy_history.append(strategy_value)
        continue

    # Build buy candidates for today
    candidates = []

    for ticker, df in all_forecasts.items():

        if date not in df.index:
            continue

        row = df.loc[date]

        for label, days in PERIODS.items():

            prob_col = f"Pred_Prob_{label}"
            std_col  = f"Pred_Prob_Std_{label}"
            act_col  = f"Actual_LogR_{label}"

            if prob_col not in row or pd.isna(row[prob_col]):
                continue

            pred_prob = float(row[prob_col])
            pred_std  = float(row.get(std_col, 0.0))
            pred_std  = max(pred_std, 1e-6)

            adj_prob = pred_prob - STD_FACTOR * pred_std

            if adj_prob <= MIN_ACCEPTED:
                continue

            actual_logr = float(row.get(act_col, np.nan))

            candidates.append({
                "Ticker"        : ticker,
                "Period"        : label,
                "Days"          : days,
                "Pred_Prob"     : pred_prob,
                "Pred_Prob_Std" : pred_std,
                "Adj_Prob"      : adj_prob,
                "Actual_LogR"   : actual_logr,
            })

    # No candidates, then nothing to buy
    if not candidates:
        strategy_history.append(strategy_value)
        continue

    # Selecting best candidates
    best = max(candidates, key=lambda x: x["Adj_Prob"])

    entry_idx = i
    exit_idx  = min(i + best["Days"], len(common_dates) - 1)

    best["entry_idx"] = entry_idx
    best["exit_idx"]  = exit_idx
    best["BuyDate"]   = date

    current_hold = best
    total_buys  += 1

    buy_points.append({
        "Date"        : date,
        "Value"       : strategy_value,
        "Ticker"      : best["Ticker"],
        "Horizon"     : best["Period"],
        "Pred_Prob"   : best["Pred_Prob"],
        "Adj_Prob"    : best["Adj_Prob"],
    })

    strategy_history.append(strategy_value)


# save trade summary
summary_df = pd.DataFrame(trade_log)
summary_df.to_csv("trade_summary_prob_strategy.csv", index=False)

print("\nSaved trade summary to trade_summary_prob_strategy.csv")
print(summary_df.head())

if total_buys > 0:
    print(f"\nBuy success rate: {successful_buys}/{total_buys} = {(successful_buys/total_buys):.2%}")
else:
    print("\nNo completed trades.")


# Random Baseline (1-day random picks)

returns_by_date = {}

for date in common_dates:
    vals = []
    for ticker, df in all_forecasts.items():
        if date in df.index and "Actual_LogR_1d" in df.columns:
            v = df.loc[date, "Actual_LogR_1d"]
            if not pd.isna(v):
                vals.append(v)
    returns_by_date[date] = vals

random_results = np.zeros((len(common_dates), random_runs))

for run in range(random_runs):
    value = initial_value
    for i, date in enumerate(common_dates):
        if returns_by_date[date]:
            pick = random.choice(returns_by_date[date])
            value *= np.exp(pick)
        random_results[i, run] = value

random_mean = np.mean(random_results, axis=1)
random_std  = np.std(random_results, axis=1)

# SPY BUY & HOLD
spy_path = PROJECT_ROOT / "TrainingData/indicators_data/processed/SPY-VIX/SPY_daily_processed.csv"

spy_df = pd.read_csv(spy_path, parse_dates=["date"])
spy_df = spy_df.rename(columns={"date": "Date", "close": "Close"})
spy_df = spy_df[spy_df["Date"].isin(common_dates)].sort_values("Date").reset_index(drop=True)
spy_df["PortfolioValue"] = initial_value * (spy_df["Close"] / spy_df["Close"].iloc[0])

def save_final_plot(
    dates, 
    random_results, 
    strat_values, 
    spy_values, 
    random_mean=None, 
    random_std=None, 
    show_uncertainty=False,
    filename="final_plot.png"
):
    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    # Plot random runs faintly
    for r in range(random_results.shape[1]):
        ax.plot(dates, random_results[:, r], alpha=0.10, lw=1, color="white")

    # Plot uncertainty shading (if enabled)
    if show_uncertainty and random_mean is not None:
        ax.fill_between(
            dates, 
            random_mean - 3 * random_std,
            random_mean + 3 * random_std,
            color="gray", alpha=0.08,
        )
        ax.fill_between(
            dates, 
            random_mean - random_std,
            random_mean + random_std,
            color="gray", alpha=0.20,
        )

    # SPY line
    ax.plot(dates, spy_values, color="white", lw=3, label="SPY")

    # Strategy line
    ax.plot(dates, strat_values, color="#39FF14", lw=3, label="Prob Strategy")

    ymin = np.nanmin(random_results)
    ymax = np.nanmax(np.concatenate([random_results.flatten(), strat_values, spy_values]))
    margin = 0.05 * (ymax - ymin)

    ax.set_ylim(ymin - margin, ymax + margin)
    ax.set_xlim(dates[0], dates[-1])

    ax.set_title("AI Probability Strategy vs Random vs SPY", color="white", fontsize=22)
    ax.set_xlabel("Date", color="white")
    ax.set_ylabel("Portfolio Value", color="white")
    ax.tick_params(colors="white")

    legend = ax.legend(facecolor="black", edgecolor="white", fontsize=12)
    for text in legend.get_texts():
        text.set_color("white")

    # Save PNG
    png_path = video_dir / filename
    plt.savefig(png_path, dpi=200, facecolor="black")
    force_png_rgb(png_path)
    plt.close(fig)

    print(f"Saved FULL STATIC PNG: {png_path}")

strategy_history_arr = np.array(strategy_history)
spy_values_arr       = spy_df["PortfolioValue"].values

save_final_plot(
    common_dates,
    random_results,
    strategy_history_arr,
    spy_values_arr,
    show_uncertainty=False,
    filename="random_vs_prob_strategy_clean.png"
)

save_final_plot(
    common_dates,
    random_results,
    strategy_history_arr,
    spy_values_arr,
    random_mean=random_mean,
    random_std=random_std,
    show_uncertainty=True,
    filename="random_vs_prob_strategy_uncertainty.png"
)
