"""
Export trained LSTM models to Pythia's artifact directory structure.

Usage:
    python export_for_pythia.py [--pythia-dir /path/to/pythia_divination]

This script:
1. Copies .keras models and feature_columns JSON files
2. Copies pre-fitted StandardScaler (from training)
3. Generates metrics.json from the model metadata
"""

import shutil
import json
import argparse
from pathlib import Path

# --- Config ---
MODELS_DIR = Path("TrainingData/models")

EXPORTS = {
    "lstm_5d": {
        "model_src": MODELS_DIR / "lstm_production.keras",
        "features_src": MODELS_DIR / "feature_columns_production.json",
        "scaler_src": MODELS_DIR / "scaler_production.joblib",
        "task": "classifier",
        "description": "5-day return predictor (>2% in 5 days)",
    },
    "lstm_jackpot": {
        "model_src": MODELS_DIR / "lstm_jackpot.keras",
        "features_src": MODELS_DIR / "feature_columns_jackpot.json",
        "scaler_src": MODELS_DIR / "scaler_jackpot.joblib",
        "task": "classifier",
        "description": "Jackpot predictor (>20% in 20 days)",
    },
}


def export_model(name: str, cfg: dict, pythia_artifacts: Path):
    """Export a single model to Pythia artifacts."""
    dest_dir = pythia_artifacts / name / cfg["task"]
    dest_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy .keras model
    if cfg["model_src"].exists():
        shutil.copy2(cfg["model_src"], dest_dir / "model.keras")
        print(f"  Copied model: {cfg['model_src']} -> {dest_dir / 'model.keras'}")
    else:
        print(f"  WARNING: Model not found: {cfg['model_src']}")
        return False

    # 2. Copy feature columns
    if cfg["features_src"].exists():
        shutil.copy2(cfg["features_src"], dest_dir / "feature_columns.json")
        print(f"  Copied features: {cfg['features_src']}")
    else:
        print(f"  WARNING: Feature columns not found: {cfg['features_src']}")

    # 3. Copy scaler
    if cfg["scaler_src"].exists():
        shutil.copy2(cfg["scaler_src"], dest_dir / "scaler.joblib")
        print(f"  Copied scaler: {cfg['scaler_src']}")
    else:
        print(f"  WARNING: Scaler not found: {cfg['scaler_src']}")
        print(f"  (Run training first to generate the scaler)")

    # 4. Generate metrics.json
    metrics = {
        "model_name": name,
        "description": cfg["description"],
        "task": cfg["task"],
        "source": "LSTM_AI_Stock_Predictor",
        "sequence_length": 60,
    }
    metrics_path = dest_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Wrote metrics: {metrics_path}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Export LSTM models to Pythia artifacts")
    parser.add_argument(
        "--pythia-dir",
        type=str,
        default="/tmp/pythia/pythia_divination",
        help="Path to Pythia's pythia_divination directory",
    )
    args = parser.parse_args()

    pythia_artifacts = Path(args.pythia_dir) / "artifacts"
    print(f"Exporting to: {pythia_artifacts}")
    print("=" * 60)

    success_count = 0
    for name, cfg in EXPORTS.items():
        print(f"\nExporting {name}...")
        if export_model(name, cfg, pythia_artifacts):
            success_count += 1

    print(f"\n{'=' * 60}")
    print(f"Export complete: {success_count}/{len(EXPORTS)} models exported.")

    if success_count == len(EXPORTS):
        print("\nTo verify, start Pythia and run:")
        print(f"  cd {args.pythia_dir} && python main.py")
        print("  curl localhost:8000/models")
        print("  curl localhost:8000/predict/AAPL?model=lstm_5d")
        print("  curl localhost:8000/predict/AAPL?model=lstm_jackpot")


if __name__ == "__main__":
    main()
