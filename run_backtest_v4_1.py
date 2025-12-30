import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- CONFIG ---
PROJECT_ROOT = Path(os.getcwd())
FORECAST_DIR = PROJECT_ROOT / "forecasts"
VIDEO_DIR    = PROJECT_ROOT / "videos"
os.makedirs(VIDEO_DIR, exist_ok=True)

INITIAL_CAPITAL = 10000.0
MAX_POSITIONS = 10         # Hold up to 10 stocks at once
POS_SIZE = 1.0 / MAX_POSITIONS # Allocate 10% per trade
MIN_PROB = 0.70            # Only buy if model is 70% sure
SELL_THRESHOLD = 0.50      # Sell if confidence drops below 50%

def main():
    print("Starting Multi-Slot Backtest...")
    
    # 1. Load all forecasts
    all_files = [f for f in os.listdir(FORECAST_DIR) if f.endswith("_forecast.csv")]
    if not all_files:
        print("No forecasts found. Run the training notebook first.")
        return

    # Consolidate into one big dataframe
    master_df = pd.DataFrame()
    for f in all_files:
        temp = pd.read_csv(FORECAST_DIR / f)
        master_df = pd.concat([master_df, temp])
    
    master_df['date'] = pd.to_datetime(master_df['date'])
    master_df = master_df.sort_values('date')
    
    # 2. Simulation Loop
    dates = master_df['date'].unique()
    cash = INITIAL_CAPITAL
    holdings = {} # {ticker: {'shares': 10, 'buy_price': 100}}
    portfolio_history = []
    
    print(f"   Simulating across {len(dates)} days...")

    for d in dates:
        # Get predictions for this specific day
        daily_preds = master_df[master_df['date'] == d]
        
        # A. Sell Logic (Check current holdings)
        # In a real backtest, we need updated prices for 'today'. 
        # Since this is a forecast file, we assume 'close' is today's price.
        
        # Convert daily_preds to look-up dict for speed
        daily_map = daily_preds.set_index('ticker').to_dict('index')
        
        tickers_to_sell = []
        for ticker, info in holdings.items():
            # If we have new data for this stock
            if ticker in daily_map:
                curr_price = daily_map[ticker]['close']
                curr_prob = daily_map[ticker]['prob_1w']
                
                # Sell if probability drops or stop loss (optional)
                if curr_prob < SELL_THRESHOLD:
                    cash += info['shares'] * curr_price
                    tickers_to_sell.append(ticker)
                    # print(f"   Sold {ticker} at ${curr_price:.2f} (Prob: {curr_prob:.2f})")
            
        for t in tickers_to_sell:
            del holdings[t]
            
        # B. Buy Logic
        # Find top picks we don't own
        # Filter for high probability
        candidates = daily_preds[daily_preds['prob_1w'] > MIN_PROB].sort_values('prob_1w', ascending=False)
        
        open_slots = MAX_POSITIONS - len(holdings)
        
        if open_slots > 0 and not candidates.empty:
            for _, row in candidates.iterrows():
                if open_slots == 0: break
                if row['ticker'] in holdings: continue
                
                # Execute Buy
                # Position Value = Total Capital * 10% (Fixed Fractional)
                # Or Current Portfolio Value * 10% (Compounding)
                # Let's use Current Portfolio Value
                current_port_val = cash + sum([h['shares'] * daily_map.get(t, {'close': h['buy_price']})['close'] for t, h in holdings.items() if t in daily_map])
                
                trade_amt = current_port_val * POS_SIZE
                if cash < trade_amt: trade_amt = cash # specific case for low cash
                
                shares = trade_amt / row['close']
                holdings[row['ticker']] = {'shares': shares, 'buy_price': row['close']}
                cash -= trade_amt
                open_slots -= 1
                # print(f"   Bought {row['ticker']} at ${row['close']:.2f} (Prob: {row['prob_1w']:.2f})")

        # C. Record Value
        # Calculate Equity
        equity = 0
        for t, info in holdings.items():
            # Use today's price if available, else last buy price (rough approx for gaps)
            price = daily_map[t]['close'] if t in daily_map else info['buy_price']
            equity += info['shares'] * price
            
        total_val = cash + equity
        portfolio_history.append({'date': d, 'value': total_val})

    # 3. Results & Plotting
    res_df = pd.DataFrame(portfolio_history)
    res_df.set_index('date', inplace=True)
    
    final_val = res_df['value'].iloc[-1]
    ret = ((final_val - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    
    print("-" * 30)
    print(f"Final Portfolio: ${final_val:,.2f}")
    print(f"Return: {ret:.2f}%")
    print("-" * 30)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.style.use('dark_background')
    plt.plot(res_df.index, res_df['value'], color='#00ff00', label='AI Strategy')
    plt.axhline(INITIAL_CAPITAL, color='white', linestyle='--', alpha=0.5)
    plt.title(f"AI Strategy Performance (Return: {ret:.2f}%)")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(color='gray', alpha=0.2)
    
    save_path = VIDEO_DIR / "backtest_result.png"
    plt.savefig(save_path)
    print(f"📸 Chart saved to {save_path}")

if __name__ == "__main__":
    main()