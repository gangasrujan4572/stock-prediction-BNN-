"""
Data Preprocessing and Feature Engineering Module
Replicating the pipeline from:
'Bayesian Deep Learning and Probabilistic Forecasting of Stock Prices'
(Algorithms 2026, 19, 391)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

def load_stock_data(csv_path: str, max_rows: int = 5187) -> pd.DataFrame:
    """
    Loads raw stock data from CSV downloaded via yfinance.
    Skips initial ticker/header rows and extracts the first `max_rows` observations (5187 observations).
    """
    raw = pd.read_csv(csv_path)
    # The yfinance multi-index CSV typically has ticker in row 0, header in row 1
    df = raw.iloc[2:2 + max_rows].copy()
    df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
    
    # Cast numerical columns to float
    for col in ["Close", "High", "Low", "Open", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.reset_index(drop=True)
    return df

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes technical indicators as specified in Section 2.2 of the paper:
    - Relative Strength Index (RSI): 14-period
    - Moving Average Convergence Divergence (MACD): fast EMA=12, slow EMA=26, signal=9
    - Simple Moving Average (SMA-20): 20-period
    - Exponential Moving Average (EMA-20): 20-period
    - Bollinger Bands: 20-period window, 2 standard deviations (BB-Upper, BB-Lower)
    - Volume EMA-10: 10-period EMA of volume (used in correlation matrix)
    """
    df = df.copy()
    
    # 1. RSI (14-period) using Wilder's exponential smoothing
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0.0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = gain / loss
    df["RSI"] = np.where(loss == 0, np.where(gain == 0, 50.0, 100.0), 100.0 - (100.0 / (1.0 + rs)))
    
    # 2. MACD (12, 26, 9)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_diff"] = df["MACD"] - df["MACD_signal"]
    
    # 3. SMA-20
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    
    # 4. EMA-20
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    
    # 5. Bollinger Bands (20-period, 2 std)
    rolling_std = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = df["SMA_20"] + 2.0 * rolling_std
    df["BB_Lower"] = df["SMA_20"] - 2.0 * rolling_std
    
    # 6. Volume EMA-10 (for correlation matrix)
    df["Volume_EMA_10"] = df["Volume"].ewm(span=10, adjust=False).mean()
    
    return df

def clean_and_prepare_dataset(csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads raw data, calculates indicators, and discards the first 20 observations
    due to rolling window computation, leaving exactly 5167 usable observations.
    Returns:
    - raw_df (5187 observations, for EDA and descriptive stats)
    - clean_df (5167 usable observations with all features)
    """
    raw_df = load_stock_data(csv_path, max_rows=5187)
    df_with_indicators = compute_technical_indicators(raw_df)
    clean_df = df_with_indicators.iloc[20:].reset_index(drop=True)
    return raw_df, clean_df

# The 11 input features used for the models (Section 2.2)
MODEL_FEATURES = [
    "Open", "High", "Low", "Close", "Volume",
    "RSI", "MACD", "SMA_20", "EMA_20", "BB_Upper", "BB_Lower"
]

def split_and_scale_data(clean_df: pd.DataFrame, train_ratio: float = 0.6, val_ratio: float = 0.2):
    """
    Splits into 60% Train, 20% Validation, 20% Test.
    Applies Z-score standardization using statistics computed exclusively on the training set.
    Returns scaled splits, scalers (mean, std), and original unscaled targets.
    """
    n = len(clean_df)
    n_train = int(n * train_ratio)      # 3100
    n_val = int(n * val_ratio)          # 1033
    n_test = n - n_train - n_val        # 1034
    
    train_df = clean_df.iloc[:n_train].copy().reset_index(drop=True)
    val_df = clean_df.iloc[n_train:n_train + n_val].copy().reset_index(drop=True)
    test_df = clean_df.iloc[n_train + n_val:].copy().reset_index(drop=True)
    
    # Compute mean and standard deviation exclusively on train set
    means = train_df[MODEL_FEATURES].mean()
    stds = train_df[MODEL_FEATURES].std()
    
    # Z-score standardization: (x - mean) / std
    train_scaled = (train_df[MODEL_FEATURES] - means) / stds
    val_scaled = (val_df[MODEL_FEATURES] - means) / stds
    test_scaled = (test_df[MODEL_FEATURES] - means) / stds
    
    scaler_info = {
        "means": means,
        "stds": stds,
        "close_mean": means["Close"],
        "close_std": stds["Close"]
    }
    
    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "train_scaled": train_scaled,
        "val_scaled": val_scaled,
        "test_scaled": test_scaled,
        "scaler_info": scaler_info,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test
    }

def create_rolling_sequences(
    scaled_features: np.ndarray,
    target_series: np.ndarray,
    window_size: int = 60
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates rolling window sequences of length 60:
    - X: (num_samples, 60, num_features)
    - y: (num_samples,) target at t+1
    """
    X, y = [], []
    for i in range(len(scaled_features) - window_size):
        X.append(scaled_features[i : i + window_size])
        y.append(target_series[i + window_size])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def generate_model_datasets(clean_df: pd.DataFrame, window_size: int = 60):
    """
    Prepares full sequence datasets for Bayesian LSTM (3D: N x 60 x 11)
    and flattened datasets for BNN and GPR (2D: N x 660).
    Uses continuous context across splits to prevent edge sample loss in test set.
    """
    splits = split_and_scale_data(clean_df)
    scaler = splits["scaler_info"]
    
    # Full standardized feature array across all usable observations
    all_scaled = (clean_df[MODEL_FEATURES] - scaler["means"]) / scaler["stds"]
    all_scaled_values = all_scaled.values
    all_close_scaled = all_scaled["Close"].values
    
    n_train = splits["n_train"]  # 3100
    n_val = splits["n_val"]      # 1033
    n_test = splits["n_test"]    # 1034
    
    # Training sequences: from index 0 to n_train
    X_train_3d, y_train = create_rolling_sequences(
        all_scaled_values[:n_train],
        all_close_scaled[:n_train],
        window_size=window_size
    )
    
    # Validation sequences: using context up to n_train+n_val
    X_val_3d, y_val = create_rolling_sequences(
        all_scaled_values[n_train - window_size : n_train + n_val],
        all_close_scaled[n_train - window_size : n_train + n_val],
        window_size=window_size
    )
    
    # Test sequences: using context up to end
    X_test_3d, y_test = create_rolling_sequences(
        all_scaled_values[n_train + n_val - window_size :],
        all_close_scaled[n_train + n_val - window_size :],
        window_size=window_size
    )
    
    # Flattened representations for BNN & GPR: (N, 60 * 11) = (N, 660)
    X_train_flat = X_train_3d.reshape(X_train_3d.shape[0], -1)
    X_val_flat = X_val_3d.reshape(X_val_3d.shape[0], -1)
    X_test_flat = X_test_3d.reshape(X_test_3d.shape[0], -1)
    
    # Test dates corresponding to the predictions
    test_dates = clean_df["Date"].iloc[n_train + n_val :].reset_index(drop=True)
    test_actual_unscaled = clean_df["Close"].iloc[n_train + n_val :].values
    val_actual_unscaled = clean_df["Close"].iloc[n_train : n_train + n_val].values
    train_actual_unscaled = clean_df["Close"].iloc[window_size : n_train].values
    
    # Last Close in each 60-step lookback window
    all_close_raw = clean_df["Close"].values
    train_last_close = all_close_raw[window_size - 1 : n_train - 1]
    val_last_close = all_close_raw[n_train - 1 : n_train + n_val - 1]
    test_last_close = all_close_raw[n_train + n_val - 1 : len(clean_df) - 1]
    
    # Stationary next-day price changes (Delta)
    train_delta = train_actual_unscaled - train_last_close
    val_delta = val_actual_unscaled - val_last_close
    test_delta = test_actual_unscaled - test_last_close
    
    delta_std = float(np.std(train_delta))
    y_train_delta = (train_delta / delta_std).astype(np.float32)
    y_val_delta = (val_delta / delta_std).astype(np.float32)
    y_test_delta = (test_delta / delta_std).astype(np.float32)
    
    return {
        "splits": splits,
        "scaler": scaler,
        "X_train_3d": X_train_3d,
        "y_train": y_train,
        "X_val_3d": X_val_3d,
        "y_val": y_val,
        "X_test_3d": X_test_3d,
        "y_test": y_test,
        "X_train_flat": X_train_flat,
        "X_val_flat": X_val_flat,
        "X_test_flat": X_test_flat,
        "test_dates": test_dates,
        "test_actual_unscaled": test_actual_unscaled,
        "val_actual_unscaled": val_actual_unscaled,
        "train_actual_unscaled": train_actual_unscaled,
        "train_last_close": train_last_close,
        "val_last_close": val_last_close,
        "test_last_close": test_last_close,
        "train_delta": train_delta,
        "val_delta": val_delta,
        "test_delta": test_delta,
        "delta_std": delta_std,
        "y_train_delta": y_train_delta,
        "y_val_delta": y_val_delta,
        "y_test_delta": y_test_delta
    }

def get_descriptive_stats(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes descriptive statistics for Close, High, Low, Open, Volume, RSI, MACD
    matching Table 2 and Table 3 in the paper.
    """
    df = compute_technical_indicators(raw_df)
    vars_to_include = ["Close", "High", "Low", "Open", "Volume", "RSI", "MACD"]
    stats_list = []
    
    for var in vars_to_include:
        s = df[var]
        stats_list.append({
            "Variable": var,
            "Count": int(s.count()),
            "Mean": s.mean(),
            "Std": s.std(),
            "Min": s.min(),
            "25%": s.quantile(0.25),
            "50%": s.quantile(0.50),
            "75%": s.quantile(0.75),
            "Max": s.max()
        })
    return pd.DataFrame(stats_list)
