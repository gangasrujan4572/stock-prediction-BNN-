"""
Visualization Module
Generates all 10 figures from the paper:
- Figure 1: FirstRand stock closing prices over time (2005-2025)
- Figure 2: Discovery stock closing prices over time (2005-2025)
- Figure 3: FirstRand correlation matrix heatmap
- Figure 4: Discovery correlation matrix heatmap
- Figure 5: FirstRand GPR forecast with 95% CI
- Figure 6: Discovery GPR forecast with 95% CI
- Figure 7: FirstRand Bayesian LSTM train/val/test forecast with 95% CI
- Figure 8: Discovery Bayesian LSTM train/val/test forecast with 95% CI
- Figure 9: FirstRand BNN forecast with 95% CI
- Figure 10: Discovery BNN forecast with 95% CI
"""

import os
from typing import Any
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np

# Set clean aesthetic style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"

def plot_historical_price(
    df: pd.DataFrame,
    company_name: str,
    output_path: str
):
    """
    Generates Figures 1 and 2: Historical closing prices over time.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    
    ax.plot(df["Date"], df["Close"], label=company_name, color="#1f77b4", linewidth=1.2)
    ax.set_title(f"{company_name} Stock Closing Prices Over Time", fontsize=12, pad=12)
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Price (ZAR)", fontsize=10)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)
    
    ax.xaxis.set_major_locator(mdates.YearLocator(4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_correlation_matrix(
    df: pd.DataFrame,
    company_name: str,
    output_path: str
):
    """
    Generates Figures 3 and 4: Correlation Matrix of float variables.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    cols = [
        "Close", "High", "Low", "Open",
        "RSI", "MACD", "MACD_signal", "MACD_diff",
        "SMA_20", "EMA_20", "BB_Upper", "BB_Lower", "Volume_EMA_10"
    ]
    labels = [
        "Close", "High", "Low", "Open",
        "rsi", "macd", "macd_signal", "macd_diff",
        "sma_20", "ema_20", "bollinger_h", "bollinger_l", "volume_ema_10"
    ]
    
    corr_df = df[cols].copy()
    corr_df.columns = labels
    corr = corr_df.corr()
    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-0.3,
        vmax=1.0,
        square=True,
        cbar_kws={"shrink": 0.8},
        ax=ax,
        annot_kws={"size": 8}
    )
    ax.set_title("Correlation Matrix of Float Variables", fontsize=12, pad=12)
    plt.xticks(rotation=90, fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_gpr_forecast(
    dates: Any,
    actual: np.ndarray,
    predicted: np.ndarray,
    lower_ci: np.ndarray,
    upper_ci: np.ndarray,
    company_name: str,
    output_path: str
):
    """
    Generates Figures 5 and 6: GPR forecast actual vs predicted with 95% CI.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dates_dt = pd.to_datetime(dates)
    fig, ax = plt.subplots(figsize=(11, 5), dpi=300)
    
    ax.plot(dates_dt, actual, label="Actual", color="black", linewidth=1.1)
    ax.plot(dates_dt, predicted, label="Predicted GPR", color="#d62728", linewidth=1.1, linestyle="-")
    ax.fill_between(dates_dt, lower_ci, upper_ci, color="#ff7f0e", alpha=0.25, label="95% Confidence Interval")
    
    ax.set_title(f"{company_name} Gaussian Process: Actual vs Predicted with 95% Confidence Interval", fontsize=12, pad=12)
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Stock Price (ZAR Cents)", fontsize=10)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_full_pipeline_forecast(
    train_dates: Any,
    train_y: np.ndarray,
    val_dates: Any,
    val_y: np.ndarray,
    test_dates: Any,
    test_y: np.ndarray,
    test_pred: np.ndarray,
    lower_ci: np.ndarray,
    upper_ci: np.ndarray,
    model_name: str,
    company_name: str,
    output_path: str
):
    """
    Generates Figures 7, 8, 9, 10:
    Train / Validation / Test price trajectory with 95% Confidence Interval.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    train_dt = pd.to_datetime(train_dates)[:len(train_y)]
    val_dt = pd.to_datetime(val_dates)[:len(val_y)]
    test_dt = pd.to_datetime(test_dates)[:len(test_y)]
    
    fig, ax = plt.subplots(figsize=(11, 5), dpi=300)
    
    ax.plot(train_dt, train_y[:len(train_dt)], label="Train (Actual)", color="#1f77b4", linewidth=1.2)
    ax.plot(val_dt, val_y[:len(val_dt)], label="Validation (Actual)", color="#2ca02c", linewidth=1.2)
    ax.plot(test_dt, test_y[:len(test_dt)], label="Test (Actual)", color="black", linewidth=1.2)
    ax.plot(test_dt, test_pred[:len(test_dt)], label=f"Test (Predicted {model_name})", color="#d62728", linewidth=1.2)
    ax.fill_between(test_dt, lower_ci[:len(test_dt)], upper_ci[:len(test_dt)], color="#ff7f0e", alpha=0.3, label="95% Confidence Interval")
    
    title = f"{company_name} {model_name}: Train / Validation / Test with 95% CI"
    ax.set_title(title, fontsize=12, pad=12)
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Close Price (ZAR Cents)", fontsize=10)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    ax.xaxis.set_major_locator(mdates.YearLocator(4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
