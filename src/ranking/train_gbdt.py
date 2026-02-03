"""
FAANG-style GBDT training pipeline for feed ranking.

This module trains a relevance model used in the Rank stage
of a multi-stage feed ranking system.

Design goals:
- Offline / online feature parity
- Deterministic training
- Calibration-ready outputs
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.isotonic import IsotonicRegression

# -----------------------------
# Configuration
# -----------------------------

DATA_PATH = Path("data/processed/train.parquet")
MODEL_PATH = Path("models/gbdt_ranker.txt")
CALIBRATOR_PATH = Path("models/calibrator.json")
RANDOM_SEED = 42

TARGET_COL = "label"

# -----------------------------
# Data Loading
# -----------------------------

def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing training data at {path}")
    return pd.read_parquet(path)

# -----------------------------
# Training
# -----------------------------

def train_gbdt(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c != TARGET_COL]

    X = df[feature_cols]
    y = df[TARGET_COL]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "seed": RANDOM_SEED,
        "verbosity": -1,
    }

    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=500,
        early_stopping_rounds=50,
        verbose_eval=50,
    )

    val_preds = model.predict(X_val)

    metrics = {
        "auc": roc_auc_score(y_val, val_preds),
        "log_loss": log_loss(y_val, val_preds),
    }

    return model, val_preds, y_val, metrics

# -----------------------------
# Calibration
# -----------------------------

def calibrate(preds: np.ndarray, labels: np.ndarray):
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(preds, labels)
    return calibrator

# -----------------------------
# Persistence
# -----------------------------

def save_model(model, calibrator):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))

    payload = {
        "thresholds": calibrator.X_thresholds_.tolist(),
        "values": calibrator.y_thresholds_.tolist(),
    }

    with open(CALIBRATOR_PATH, "w") as f:
        json.dump(payload, f, indent=2)

# -----------------------------
# Entry Point
# -----------------------------

if __name__ == "__main__":
    df = load_data(DATA_PATH)
    model, preds, labels, metrics = train_gbdt(df)
    calibrator = calibrate(preds, labels)
    save_model(model, calibrator)

    print("Training complete")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
