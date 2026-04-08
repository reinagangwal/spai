"""
SPENDLY - Forecasting Module
============================
Trains an LSTM model on normalized weekly spending data,
evaluates performance, and generates multi-step future forecasts.
 
Usage:
    python src/forecast.py
 
Expects:
    data/processed/train.csv  — weekly pivot table
                                index: date (weekly)
                                columns: exp_type categories
                                values: amount_norm
"""
 
from importlib.resources import path
from os import path
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error
 
warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")
 
# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
TRAIN_PATH   = "data/processed/train.csv"
OUTPUT_DIR   = "outputs"
FORECAST_CSV = os.path.join(OUTPUT_DIR, "forecast.csv")
LOSS_PLOT    = os.path.join(OUTPUT_DIR, "training_loss.png")
 
WINDOW_SIZE    = 4       # sliding window (weeks)
FORECAST_STEPS = 7       # weeks to forecast ahead
EPOCHS         = 20
BATCH_SIZE     = 16
VAL_SPLIT      = 0.15    # fraction of training data used for validation
LSTM_UNITS     = 64
DROPOUT_RATE   = 0.2
RANDOM_SEED    = 42
 
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
 
 
# ──────────────────────────────────────────────
# 1. DATA LOADING & SEQUENCE CREATION
# ──────────────────────────────────────────────
 
def load_pivot(path: str) -> pd.DataFrame:
    """Load raw transaction CSV and convert to weekly pivot."""

    df = pd.read_csv(path)

    # convert date
    df['date'] = pd.to_datetime(df['date'])

    # group weekly
    weekly = df.groupby(
        [pd.Grouper(key='date', freq='W'), 'exp_type']
    )['amount_norm'].sum().reset_index()

    # pivot
    df = weekly.pivot(index='date', columns='exp_type', values='amount_norm')

    # clean
    df = df.fillna(0.0)
    df = df.sort_index()

    print(f"[data] Loaded {path} → shape {df.shape}")
    print(f"[data] Categories: {list(df.columns)}")

    return df
 
 
def make_sequences(data: np.ndarray, window: int):
    """
    Sliding-window sequence builder.
 
    Parameters
    ----------
    data   : 2-D array  (timesteps, num_categories)
    window : int        look-back length
 
    Returns
    -------
    X : (samples, window, num_categories)
    y : (samples, num_categories)
    """
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i : i + window])
        y.append(data[i + window])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
 
 
# ──────────────────────────────────────────────
# 2. MODEL DEFINITION
# ──────────────────────────────────────────────
 
def build_model(window_size: int, num_categories: int) -> tf.keras.Model:
    """
    Build a simple LSTM regressor.
 
    Architecture
    ------------
    LSTM(64)  →  Dropout  →  Dense(32, relu)  →  Dense(num_categories)
    """
    model = Sequential([
        LSTM(LSTM_UNITS,
             input_shape=(window_size, num_categories),
             return_sequences=False,
             name="lstm_1"),
        Dropout(DROPOUT_RATE, name="dropout_1"),
        Dense(32, activation="relu", name="dense_hidden"),
        Dense(num_categories, activation="linear", name="output"),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.summary()
    return model
 
 
# ──────────────────────────────────────────────
# 3. TRAINING
# ──────────────────────────────────────────────
 
def train_model(model, X_train, y_train):
    """
    Train the LSTM with early stopping and return the history object.
    """
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )
 
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VAL_SPLIT,
        callbacks=[early_stop],
        verbose=1,
    )
    return history
 
 
def plot_loss(history, save_path: str):
    """Save a training vs. validation loss plot."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"],     label="Train loss")
    plt.plot(history.history["val_loss"], label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("SPENDLY – LSTM Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"[plot]   Loss curve saved → {save_path!r}")
 
 
# ──────────────────────────────────────────────
# 4. EVALUATION
# ──────────────────────────────────────────────
 
def mape_safe(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Mean Absolute Percentage Error, per column.
    Rows where y_true == 0 are excluded to avoid division by zero.
    Returns NaN for a category that is all zeros.
    """
    results = []
    for col in range(y_true.shape[1]):
        mask = y_true[:, col] != 0
        if mask.sum() == 0:
            results.append(float("nan"))
        else:
            pct_err = np.abs(
                (y_true[mask, col] - y_pred[mask, col]) / y_true[mask, col]
            )
            results.append(pct_err.mean() * 100)
    return np.array(results)
 
 
def evaluate(model, X, y_true, categories: list):
    """
    Compute and print MAE, RMSE, MAPE per category and overall.
    """
    y_pred = model.predict(X, verbose=0)
 
    mae_per  = mean_absolute_error(y_true, y_pred, multioutput="raw_values")
    rmse_per = np.sqrt(mean_squared_error(y_true, y_pred, multioutput="raw_values"))
    mape_per = mape_safe(y_true, y_pred)
 
    mae_overall  = mae_per.mean()
    rmse_overall = rmse_per.mean()
    mape_overall = np.nanmean(mape_per)
 
    # Pretty-print per-category metrics
    header = f"{'Category':<14} {'MAE':>8} {'RMSE':>8} {'MAPE (%)':>10}"
    print("\n" + "=" * len(header))
    print("EVALUATION METRICS")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for i, cat in enumerate(categories):
        mape_str = f"{mape_per[i]:>10.2f}" if not np.isnan(mape_per[i]) else f"{'N/A':>10}"
        print(f"{cat:<14} {mae_per[i]:>8.4f} {rmse_per[i]:>8.4f} {mape_str}")
    print("-" * len(header))
    print(f"{'OVERALL':<14} {mae_overall:>8.4f} {rmse_overall:>8.4f} {mape_overall:>10.2f}")
    print("=" * len(header) + "\n")
 
    return y_pred
 
 
# ──────────────────────────────────────────────
# 5. RECURSIVE FUTURE FORECASTING
# ──────────────────────────────────────────────
 
def recursive_forecast(model, last_window: np.ndarray, steps: int) -> np.ndarray:
    """
    Recursively generate `steps` future predictions.
 
    Parameters
    ----------
    last_window : (window_size, num_categories)  — the most recent context
    steps       : int                            — how many steps ahead
 
    Returns
    -------
    forecasts : (steps, num_categories)
    """
    window = last_window.copy().astype(np.float32)   # (W, C)
    forecasts = []
 
    for _ in range(steps):
        inp   = window[np.newaxis, :, :]              # (1, W, C)
        pred  = model.predict(inp, verbose=0)[0]      # (C,)
        forecasts.append(pred)
        # slide the window forward: drop oldest, append new prediction
        window = np.vstack([window[1:], pred])
 
    return np.array(forecasts)
 
 
# ──────────────────────────────────────────────
# 6. SAVE FORECAST CSV
# ──────────────────────────────────────────────
 
def save_forecast(forecasts: np.ndarray,
                  categories: list,
                  last_date: pd.Timestamp,
                  save_path: str):
    """
    Save the future forecasts to a CSV with a proper date index.
    Dates are generated weekly starting the week after the last known date.
    """
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(weeks=1),
        periods=len(forecasts),
        freq="W",
    )
    df_out = pd.DataFrame(forecasts, index=future_dates, columns=categories)
    df_out.index.name = "date"
 
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_out.to_csv(save_path)
    print(f"[save]   Forecast CSV saved → {save_path!r}")
    print(df_out.round(4).to_string())
 
 
# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
 
def main():
    # ── 1. Load data ──────────────────────────
    df = load_pivot(TRAIN_PATH)
    categories    = list(df.columns)
    num_categories = len(categories)
    data_np       = df.values.astype(np.float32)   # (T, C)
 
    # ── 2. Build sequences ────────────────────
    X, y = make_sequences(data_np, WINDOW_SIZE)
    print(f"[seq]    X shape: {X.shape}   y shape: {y.shape}")
 
    if len(X) < 10:
        raise ValueError(
            f"Not enough data to build sequences. "
            f"Need at least {WINDOW_SIZE + 10} weekly rows, got {len(data_np)}."
        )
 
    # ── 3. Build & train model ────────────────
    model   = build_model(WINDOW_SIZE, num_categories)
    history = train_model(model, X, y)
    plot_loss(history, LOSS_PLOT)
 
    # ── 4. Evaluate on full training sequences ─
    # (In production, pass held-out X_val / X_test instead)
    print("[eval]   Evaluating on training sequences …")
    evaluate(model, X, y, categories)
 
    # ── 5. Recursive forecast ─────────────────
    last_window = data_np[-WINDOW_SIZE:]            # most recent 4 weeks
    print(f"\n[forecast] Generating {FORECAST_STEPS}-step recursive forecast …")
    forecasts = recursive_forecast(model, last_window, FORECAST_STEPS)
 
    # ── 6. Save results ───────────────────────
    save_forecast(forecasts, categories, df.index[-1], FORECAST_CSV)
    print("\n[done]   SPENDLY forecasting complete.")
 
 
if __name__ == "__main__":
    main()