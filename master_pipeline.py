import subprocess
import os
import sys
import time
from datetime import datetime

# --- CONFIGURATION ---
PYTHON_EXEC = sys.executable
SCRIPTS_DIR = "TrainingData/scripts"

# ENHANCED PIPELINE with Options Flow
PIPELINE = [
    # --- PHASE 1: DATA COLLECTION ---
    ("TrainingData/featuresPy/markets.py", "Fetching Global Macro Data (VIX, Oil, Rates)"),
    ("TrainingData/featuresPy/stockScrapper.py", "Smart-Updating Stock Prices (OHLCV)"),
    ("options_flow.py", "Scraping Options Flow (Smart Money)"),
    
    # --- PHASE 2: PROCESSING ---
    ("TrainingData/featuresPy/sentiment.py", "Updating News Sentiment (FinBERT)"),
    #("TrainingData/processor.py", "Engineering Advanced Features (Options + Squeeze)"),
    
    # --- PHASE 3: TRAINING (The Gym) ---
    # Comment these out if you are happy with your current models and just want to trade!
    ("train_model.py", "Training Production Model (Target: >2% in 5 Days)"),
    ("train_jackpot_model.py", "Training Jackpot Model (Target: >20% in 20 Days)"),
]

def run_script(script_name, description):
    """Executes a single python script and waits for it to finish."""
    # Resolve Path
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    if not os.path.exists(script_path):
        script_path = script_name
    
    if not os.path.exists(script_path):
        print(f"ERROR: Could not find {script_name}")
        return False

    # Execute
    print(f"\n{'='*60}")
    print(f"STARTING: {description}")
    print(f"   [{datetime.now().strftime('%H:%M:%S')}] Executing {script_name}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    try:
        subprocess.run([PYTHON_EXEC, script_path], check=True)
        
        duration = time.time() - start_time
        print(f"\n✓ FINISHED: {script_name} (Took {duration:.1f}s)")
        return True
        
    except subprocess.CalledProcessError:
        print(f"\n✗ CRITICAL FAILURE in {script_name}. Pipeline stopped.")
        return False
    except KeyboardInterrupt:
        print("\n\n⚠ Pipeline stopped by user.")
        return False

def run_inference_dual_mode():
    """Runs daily_picks.py twice with different flags to generate both reports."""
    print(f"\n{'='*60}")
    print("PHASE 4: DUAL-MODE INFERENCE")
    print(f"{'='*60}")
    
    daily_picks_path = os.path.join(SCRIPTS_DIR, "daily_picks.py")
    if not os.path.exists(daily_picks_path):
        daily_picks_path = "daily_picks.py"

    # 1. Consistency Mode
    print("\n--- MODE 1: CONSISTENCY (Safe Swings) ---")
    try:
        subprocess.run([PYTHON_EXEC, daily_picks_path, "--mode", "consistency"], check=True)
    except subprocess.CalledProcessError:
        print("Consistency Inference Failed.")

    # 2. Jackpot Mode
    print("\n--- MODE 2: JACKPOT (High Risk/Reward) ---")
    try:
        subprocess.run([PYTHON_EXEC, daily_picks_path, "--mode", "jackpot"], check=True)
    except subprocess.CalledProcessError:
        print("Jackpot Inference Failed.")

def run_auto_trader():
    """Executes the Alpaca Trader to submit orders based on the generated reports."""
    print(f"\n{'='*60}")
    print("PHASE 5: EXECUTION (Alpaca)")
    print(f"{'='*60}")
    
    success = run_script("alpaca_trader.py", "Submitting Orders to Market")
    if success:
        print("\n✓ TRADING CYCLE COMPLETE. Check Alpaca Dashboard.")
    else:
        print("\n✗ Trading script failed. No orders placed.")

def print_banner():
    """Print startup banner with feature summary."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                 ALGO-TRADER: MASTER PIPELINE V2              ║
    ║                    (Enhanced with Options Flow)              ║
    ╚══════════════════════════════════════════════════════════════╝
    
    📊 Data Sources:
       ✓ Stock Prices (OHLCV)
       ✓ News Sentiment (FinBERT)
       ✓ Fundamentals (Finviz)
       ✓ Global Context (VIX, Commodities)
       ✓ Insider Trading (SEC)
       Options Flow (Put/Call, UOA)
    
    Feature Engineering:
       ✓ Technical Indicators (RSI, MACD, BB, ATR)
       ✓ Jackpot Features (Squeeze, RVOL, Dist-to-High)
       Short Squeeze Score
       Enhanced Momentum (multi-timeframe)
       Volume Profile (buy/sell pressure)
       Options Sentiment
    
    Models:
       ✓ Production Model (>2% in 5 days)
       ✓ Jackpot Model (>20% in 20 days)
    """
    print(banner)
    print(f"    Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"    System: {PYTHON_EXEC}")
    print(f"\n{'='*60}\n")

def main():
    print_banner()
    
    total_start = time.time()
    
    # 1. Run Data/Training Sequence
    for script, desc in PIPELINE:
        success = run_script(script, desc)
        if not success:
            print("\n⚠ PIPELINE ABORTED DUE TO ERRORS.")
            return

    # 2. Run Inference (Generate the CSVs)
    run_inference_dual_mode()
    
    # 3. Run Execution (Place the Trades)
    run_auto_trader()
    
    total_time = (time.time() - total_start) / 60
    
    print(f"\n{'='*60}")
    print(f"✓ SYSTEM SHUTDOWN COMPLETE")
    print(f"   Total Runtime: {total_time:.1f} minutes")
    print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
