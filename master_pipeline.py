import subprocess
import os
import sys
import time
from datetime import datetime

# --- CONFIGURATION ---
PYTHON_EXEC = sys.executable  # Uses the current Python environment (Colab or Local)
SCRIPTS_DIR = "TrainingData/scripts"  # Folder where your .py files live

# 1. The Data & Training Pipeline
# (Script Name, Description)
PIPELINE = [
    # --- PHASE 1: DATA COLLECTION (The Eyes) ---
    ("TrainingData/featuresPy/markets.py", "Fetching Global Macro Data (VIX, Oil, Rates)"),
    ("TrainingData/featuresPy/stockScrapper.py", "Smart-Updating Stock Prices (OHLCV)"),
    
    # --- PHASE 2: PROCESSING (The Cortex) ---
    ("TrainingData/featuresPy/sentiment.py", "Updating News Sentiment (FinBERT)"),
    ("TrainingData/processor.py", "Engineering Features (RVOL, Squeeze, Jackpot Factors)"),
    
    # --- PHASE 3: TRAINING (The Gym) ---
    # Comment these out if you are happy with your current models and just want to trade!
    ("train_model.py", "Training Production Model (Target: >2% in 5 Days)"),
    ("train_jackpot_model.py", "Training Jackpot Model (Target: >20% in 20 Days)"),
]

def run_script(script_name, description):
    """
    Executes a single python script and waits for it to finish.
    """
    # 1. Resolve Path
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    if not os.path.exists(script_path):
        # Fallback: Check root directory
        script_path = script_name
    
    if not os.path.exists(script_path):
        print(f"ERROR: Could not find {script_name} in {SCRIPTS_DIR} or Root.")
        return False

    # 2. Execute
    print(f"\nSTARTING: {description}...")
    print(f"   [{datetime.now().strftime('%H:%M:%S')}] Executing {script_name}")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        # check=True raises CalledProcessError if the script crashes
        subprocess.run([PYTHON_EXEC, script_path], check=True)
        
        duration = time.time() - start_time
        print(f"FINISHED: {script_name} (Took {duration:.1f}s)")
        return True
        
    except subprocess.CalledProcessError:
        print(f"CRITICAL FAILURE in {script_name}. Pipeline stopped.")
        return False
    except KeyboardInterrupt:
        print("\nPipeline stopped by user.")
        return False

def run_inference_dual_mode():
    """
    Runs daily_picks.py twice with different flags to generate both reports.
    """
    print("\n" + "="*40)
    print("PHASE 4: DUAL-MODE INFERENCE")
    print("="*40)
    
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
    """
    Executes the Alpaca Trader to submit orders based on the generated reports.
    """
    print("\n" + "="*40)
    print("PHASE 5: EXECUTION (Alpaca)")
    print("="*40)
    
    success = run_script("/content/drive/MyDrive/Colab Notebooks/LSTM_AI_Stock_Predictor/TrainingData/scripts/alpaca_trader.py", "Submitting Orders to Market")
    if success:
        print("\nTRADING CYCLE COMPLETE. Check Alpaca Dashboard.")
    else:
        print("\nTrading script failed. No orders placed.")

def main():
    print("==========================================")
    print("   ALGO-TRADER: MASTER PIPELINE")
    print("==========================================")
    print(f"   Date: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"   System: {PYTHON_EXEC}")
    
    total_start = time.time()
    
    # 1. Run Data/Training Sequence
    for script, desc in PIPELINE:
        success = run_script(script, desc)
        if not success:
            print("\nPIPELINE ABORTED DUE TO ERRORS.")
            return

    # 2. Run Inference (Generate the CSVs)
    run_inference_dual_mode()
    
    # 3. Run Execution (Place the Trades)
    run_auto_trader()
    
    total_time = (time.time() - total_start) / 60
    print(f"\nSYSTEM SHUTDOWN. Total Runtime: {total_time:.1f} minutes.")

if __name__ == "__main__":
    main()
