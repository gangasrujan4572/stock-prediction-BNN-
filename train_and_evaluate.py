"""
Comprehensive End-to-End Training and Evaluation Script (Tuned Reproduction)
Paper: 'Bayesian Deep Learning and Probabilistic Forecasting of Stock Prices'
(Algorithms 2026, 19, 391)

Models:
1. Bayesian Neural Network (BNN via Bayes by Backprop, ELBO, S=100)
2. Bayesian LSTM (Monte Carlo Dropout p=0.2, S=100)
3. Standard LSTM (Deterministic Baseline)
4. Gaussian Process Regression (GPR with RBF + WhiteKernel)
5. ARIMA(1, 1, 1) Baseline with 1-step walk-forward rolling forecast

Features & Targets:
- 11 Features: OHLCV, RSI, MACD, SMA-20, EMA-20, BB-Upper, BB-Lower (standardized on train set)
- Stationary delta target with exact causal price reconstruction: y_hat = Close_t + Delta_hat
- Uncertainty intervals calibrated with validation residual noise to achieve calibrated ~0.95 PICP.
- Multi-seed evaluation across 5 random seeds [42, 43, 44, 45, 46] reporting mean and std.

Usage:
    python train_and_evaluate.py --all
    python train_and_evaluate.py --company FirstRand
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

import data_processing as dp
import models
import metrics
import plots

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

def set_seeds(seed: int = 42):
    """Sets deterministic seeds across Python, NumPy, and PyTorch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_bnn_with_val(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 0.001,
    dropout: float = 0.1,
    kl_weight: float = 0.001,
    prior_std: float = 1.0,
    patience: int = 25,
    device: str = "cpu"
) -> models.BayesianNeuralNetwork:
    """
    Trains BNN using Bayes by Backprop with ELBO loss and validation early stopping.
    """
    in_features = X_train.shape[1]
    net = models.BayesianNeuralNetwork(in_features=in_features, dropout_rate=dropout, prior_std=prior_std).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)
    
    for epoch in range(1, epochs + 1):
        net.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = net(bx, sample=True).squeeze(-1)
            mse_loss = nn.functional.mse_loss(pred, by, reduction="mean")
            kl_loss = net.total_kl() / len(X_train)
            loss = mse_loss + kl_weight * kl_loss
            loss.backward()
            optimizer.step()
            
        net.eval()
        with torch.no_grad():
            val_pred = net(X_val_t, sample=False).squeeze(-1)
            v_loss = nn.functional.mse_loss(val_pred, y_val_t).item()
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_state = {k: v.cpu() for k, v in net.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
                    
    if best_state is not None:
        net.load_state_dict(best_state)
    return net

def train_bayesian_lstm_with_val(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 0.001,
    dropout: float = 0.2,
    hidden_units: int = 50,
    patience: int = 20,
    device: str = "cpu"
) -> models.BayesianLSTM:
    """
    Trains Bayesian LSTM with MC Dropout and validation early stopping.
    """
    input_size = X_train.shape[2]
    net = models.BayesianLSTM(input_size=input_size, hidden_units=hidden_units, dropout_rate=dropout).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.SmoothL1Loss(beta=0.5)
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)
    
    for epoch in range(1, epochs + 1):
        net.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = net(bx, force_dropout=True).squeeze(-1)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            
        net.eval()
        with torch.no_grad():
            val_pred = net(X_val_t, force_dropout=False).squeeze(-1)
            v_loss = criterion(val_pred, y_val_t).item()
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_state = {k: v.cpu() for k, v in net.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
                    
    if best_state is not None:
        net.load_state_dict(best_state)
    return net

def train_standard_lstm_with_val(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 0.001,
    dropout: float = 0.2,
    hidden_units: int = 50,
    patience: int = 20,
    device: str = "cpu"
) -> models.StandardLSTM:
    """
    Trains Standard Deterministic LSTM baseline with validation early stopping.
    """
    input_size = X_train.shape[2]
    net = models.StandardLSTM(input_size=input_size, hidden_units=hidden_units, dropout_rate=dropout).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.SmoothL1Loss(beta=0.5)
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)
    
    for epoch in range(1, epochs + 1):
        net.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = net(bx).squeeze(-1)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            
        net.eval()
        with torch.no_grad():
            val_pred = net(X_val_t).squeeze(-1)
            v_loss = criterion(val_pred, y_val_t).item()
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_state = {k: v.cpu() for k, v in net.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
                    
    if best_state is not None:
        net.load_state_dict(best_state)
    return net

def train_gpr(X_train: np.ndarray, y_train: np.ndarray, n_restarts: int = 2) -> models.GPRModel:
    """
    Fits Gaussian Process Regression using RBF + WhiteKernel (Section 2.5).
    """
    gpr = models.GPRModel(length_scale=10.0, noise_level=1e-3, n_restarts=n_restarts)
    gpr.fit(X_train, y_train)
    return gpr

def run_evaluation_for_company(
    company: str = "FirstRand",
    epochs_bnn: int = 200,
    epochs_lstm: int = 100,
    s_samples: int = 100,
    n_restarts_gpr: int = 2,
    seeds: list = None
):
    if seeds is None:
        seeds = SEEDS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    csv_file = f"{company}.csv"
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Could not find dataset file {csv_file}")
        
    print(f"\n{'='*80}\nTRAINING AND EVALUATING ALL MODELS (TUNED) ON: {company.upper()}\n{'='*80}")
    
    raw_df, clean_df = dp.clean_and_prepare_dataset(csv_file)
    ds = dp.generate_model_datasets(clean_df)
    
    test_actual = ds["test_actual_unscaled"]
    test_dates = ds["test_dates"]
    val_actual = ds["val_actual_unscaled"]
    train_actual = ds["train_actual_unscaled"]
    
    val_last_close = ds["val_last_close"]
    test_last_close = ds["test_last_close"]
    delta_std = ds["delta_std"]
    
    formatted_dates = test_dates.dt.strftime("%Y-%m-%d").values if hasattr(test_dates.iloc[0], 'strftime') else [str(d)[:10] for d in test_dates.values]
    train_dates_str = clean_df["Date"].iloc[:ds["splits"]["n_train"]].dt.strftime("%Y-%m-%d").values
    val_dates_str = clean_df["Date"].iloc[ds["splits"]["n_train"]:ds["splits"]["n_train"] + ds["splits"]["n_val"]].dt.strftime("%Y-%m-%d").values
    
    # Track metrics across seeds
    model_names = ["GPR", "Bayesian LSTM", "BNN", "Standard LSTM", "ARIMA"]
    seed_metrics = {m: [] for m in model_names}
    
    # Store predictions for the primary seed (42) for plotting and table export
    primary_predictions = {}
    
    # --------------------------------------------------------------------------
    # Model 5: ARIMA(1, 1, 1) Rolling Forecast (Deterministic across seeds)
    # --------------------------------------------------------------------------
    print(f"--> [ARIMA] Running 1-step walk-forward rolling ARIMA(1, 1, 1)...", flush=True)
    arima = models.ARIMABaseline(order=(1, 1, 1))
    full_series = clean_df["Close"].values
    n_train = ds["splits"]["n_train"]
    n_val = ds["splits"]["n_val"]
    n_test = ds["splits"]["n_test"]
    arima_pred = arima.fit_and_predict_rolling(full_series, train_len=n_train + n_val, eval_len=n_test)
    arima_res = metrics.compute_point_metrics(test_actual, arima_pred)
    for _ in seeds:
        seed_metrics["ARIMA"].append(arima_res)
    primary_predictions["arima_pred"] = arima_pred
    
    # --------------------------------------------------------------------------
    # Model 4: GPR (Fitted once with random_state=42)
    # --------------------------------------------------------------------------
    print(f"--> [GPR] Fitting GPR on Delta target...", flush=True)
    gpr = train_gpr(ds["X_train_flat"], ds["y_train_delta"], n_restarts=n_restarts_gpr)
    # Validation prediction to calibrate residual noise
    gpr_val_mean_z, gpr_val_std_z = gpr.predict(ds["X_val_flat"])
    gpr_val_pred = val_last_close + gpr_val_mean_z * delta_std
    gpr_val_res_std = np.std(val_actual - gpr_val_pred)
    
    # Test prediction
    gpr_test_mean_z, gpr_test_std_z = gpr.predict(ds["X_test_flat"])
    gpr_pred = test_last_close + gpr_test_mean_z * delta_std
    gpr_std_epistemic = gpr_test_std_z * delta_std
    gpr_std_total = np.sqrt(gpr_std_epistemic**2 + gpr_val_res_std**2)
    gpr_lower = gpr_pred - 1.96 * gpr_std_total
    gpr_upper = gpr_pred + 1.96 * gpr_std_total
    
    gpr_res = metrics.evaluate_all(test_actual, gpr_pred, gpr_lower, gpr_upper)
    for _ in seeds:
        seed_metrics["GPR"].append(gpr_res)
    primary_predictions["gpr_pred"] = gpr_pred
    primary_predictions["gpr_std"] = gpr_std_total
    primary_predictions["gpr_lower"] = gpr_lower
    primary_predictions["gpr_upper"] = gpr_upper

    # --------------------------------------------------------------------------
    # Multi-seed Loop for Neural Models (BNN, Bayesian LSTM, Standard LSTM)
    # --------------------------------------------------------------------------
    for seed_idx, seed in enumerate(seeds):
        set_seeds(seed)
        print(f"\n--> Running Seed {seed_idx + 1}/{len(seeds)} (seed={seed})...", flush=True)
        
        # 1. BNN (BBB)
        bnn = train_bnn_with_val(
            ds["X_train_flat"], ds["y_train_delta"],
            ds["X_val_flat"], ds["y_val_delta"],
            epochs=epochs_bnn, batch_size=32, lr=0.001, dropout=0.1, kl_weight=0.001,
            device=device
        )
        bnn_val_z, _ = bnn.predict_monte_carlo(ds["X_val_flat"], s_samples=s_samples, device=device)
        bnn_val_pred = val_last_close + bnn_val_z * delta_std
        bnn_val_res_std = np.std(val_actual - bnn_val_pred)
        
        bnn_test_z, bnn_test_std_z = bnn.predict_monte_carlo(ds["X_test_flat"], s_samples=s_samples, device=device)
        bnn_pred = test_last_close + bnn_test_z * delta_std
        bnn_std_epistemic = bnn_test_std_z * delta_std
        bnn_std_total = np.sqrt(bnn_std_epistemic**2 + bnn_val_res_std**2)
        bnn_lower = bnn_pred - 1.96 * bnn_std_total
        bnn_upper = bnn_pred + 1.96 * bnn_std_total
        bnn_res = metrics.evaluate_all(test_actual, bnn_pred, bnn_lower, bnn_upper)
        seed_metrics["BNN"].append(bnn_res)
        
        # 2. Bayesian LSTM (MC Dropout)
        blstm = train_bayesian_lstm_with_val(
            ds["X_train_3d"], ds["y_train_delta"],
            ds["X_val_3d"], ds["y_val_delta"],
            epochs=epochs_lstm, batch_size=32, lr=0.001, dropout=0.2, hidden_units=50,
            device=device
        )
        blstm_val_z, _ = blstm.predict_monte_carlo(ds["X_val_3d"], s_samples=s_samples, device=device)
        blstm_val_pred = val_last_close + blstm_val_z * delta_std
        blstm_val_res_std = np.std(val_actual - blstm_val_pred)
        
        blstm_test_z, blstm_test_std_z = blstm.predict_monte_carlo(ds["X_test_3d"], s_samples=s_samples, device=device)
        blstm_pred = test_last_close + blstm_test_z * delta_std
        blstm_std_epistemic = blstm_test_std_z * delta_std
        blstm_std_total = np.sqrt(blstm_std_epistemic**2 + blstm_val_res_std**2)
        blstm_lower = blstm_pred - 1.96 * blstm_std_total
        blstm_upper = blstm_pred + 1.96 * blstm_std_total
        blstm_res = metrics.evaluate_all(test_actual, blstm_pred, blstm_lower, blstm_upper)
        seed_metrics["Bayesian LSTM"].append(blstm_res)
        
        # 3. Standard LSTM
        std_lstm = train_standard_lstm_with_val(
            ds["X_train_3d"], ds["y_train_delta"],
            ds["X_val_3d"], ds["y_val_delta"],
            epochs=epochs_lstm, batch_size=32, lr=0.001, dropout=0.2, hidden_units=50,
            device=device
        )
        std_lstm_test_z = std_lstm.predict(ds["X_test_3d"], device=device)
        std_lstm_pred = test_last_close + std_lstm_test_z * delta_std
        std_lstm_res = metrics.compute_point_metrics(test_actual, std_lstm_pred)
        seed_metrics["Standard LSTM"].append(std_lstm_res)
        
        if seed == seeds[0]:
            primary_predictions["bnn_pred"] = bnn_pred
            primary_predictions["bnn_std"] = bnn_std_total
            primary_predictions["bnn_lower"] = bnn_lower
            primary_predictions["bnn_upper"] = bnn_upper
            
            primary_predictions["bayesian_lstm_pred"] = blstm_pred
            primary_predictions["bayesian_lstm_std"] = blstm_std_total
            primary_predictions["bayesian_lstm_lower"] = blstm_lower
            primary_predictions["bayesian_lstm_upper"] = blstm_upper
            
            primary_predictions["standard_lstm_pred"] = std_lstm_pred

    # Compute mean and std summary across seeds
    summary_rows = []
    for m in model_names:
        records = seed_metrics[m]
        maes = [r["MAE"] for r in records]
        rmses = [r["RMSE"] for r in records]
        mses = [r["MSE"] for r in records]
        picps = [r["PICP"] for r in records if "PICP" in r]
        mpiws = [r["MPIW"] for r in records if "MPIW" in r]
        
        row = {
            "Model": m,
            "MAE_Mean": np.mean(maes),
            "MAE_Std": np.std(maes),
            "RMSE_Mean": np.mean(rmses),
            "RMSE_Std": np.std(rmses),
            "MSE_Mean": np.mean(mses),
            "MSE_Std": np.std(mses),
            "PICP_Mean": np.mean(picps) if picps else np.nan,
            "PICP_Std": np.std(picps) if picps else np.nan,
            "MPIW_Mean": np.mean(mpiws) if mpiws else np.nan,
            "MPIW_Std": np.std(mpiws) if mpiws else np.nan,
            "MAE": np.mean(maes),
            "RMSE": np.mean(rmses),
            "MSE": np.mean(mses),
            "PICP": np.mean(picps) if picps else np.nan,
            "MPIW": np.mean(mpiws) if mpiws else np.nan
        }
        summary_rows.append(row)
        
    summary_df = pd.DataFrame(summary_rows)
    print(f"\n{'='*80}\nFINAL MULTI-SEED (N=5) TEST RESULTS SUMMARY: {company}\n{'='*80}")
    print(summary_df[["Model", "MAE_Mean", "MAE_Std", "RMSE_Mean", "RMSE_Std", "PICP_Mean", "MPIW_Mean"]].to_string(index=False))
    
    # Save predictions .npz file (using representative primary seed predictions)
    npz_path = os.path.join(OUTPUT_DIR, f"{company}_predictions.npz")
    np.savez_compressed(
        npz_path,
        company=company,
        test_dates=np.array(formatted_dates),
        test_actual=test_actual,
        train_dates=np.array(train_dates_str),
        train_actual=train_actual,
        val_dates=np.array(val_dates_str),
        val_actual=val_actual,
        bnn_pred=primary_predictions["bnn_pred"],
        bnn_std=primary_predictions["bnn_std"],
        bnn_lower=primary_predictions["bnn_lower"],
        bnn_upper=primary_predictions["bnn_upper"],
        bayesian_lstm_pred=primary_predictions["bayesian_lstm_pred"],
        bayesian_lstm_std=primary_predictions["bayesian_lstm_std"],
        bayesian_lstm_lower=primary_predictions["bayesian_lstm_lower"],
        bayesian_lstm_upper=primary_predictions["bayesian_lstm_upper"],
        gpr_pred=primary_predictions["gpr_pred"],
        gpr_std=primary_predictions["gpr_std"],
        gpr_lower=primary_predictions["gpr_lower"],
        gpr_upper=primary_predictions["gpr_upper"],
        standard_lstm_pred=primary_predictions["standard_lstm_pred"],
        arima_pred=primary_predictions["arima_pred"]
    )
    print(f"Saved predictions to: {npz_path}")
    
    # Save predictions CSV
    pred_df = pd.DataFrame({
        "Date": formatted_dates,
        "Actual_Close": test_actual,
        "BNN_Pred": primary_predictions["bnn_pred"],
        "BNN_CI_Lower": primary_predictions["bnn_lower"],
        "BNN_CI_Upper": primary_predictions["bnn_upper"],
        "Bayesian_LSTM_Pred": primary_predictions["bayesian_lstm_pred"],
        "Bayesian_LSTM_CI_Lower": primary_predictions["bayesian_lstm_lower"],
        "Bayesian_LSTM_CI_Upper": primary_predictions["bayesian_lstm_upper"],
        "GPR_Pred": primary_predictions["gpr_pred"],
        "GPR_CI_Lower": primary_predictions["gpr_lower"],
        "GPR_CI_Upper": primary_predictions["gpr_upper"],
        "Standard_LSTM_Pred": primary_predictions["standard_lstm_pred"],
        "ARIMA_Pred": primary_predictions["arima_pred"]
    })
    pred_csv_path = os.path.join(OUTPUT_DIR, f"Trained_Predictions_{company}.csv")
    pred_df.to_csv(pred_csv_path, index=False)
    
    # Save multi-seed metrics table
    metrics_csv_path = os.path.join(OUTPUT_DIR, f"Trained_Metrics_{company}.csv")
    summary_df.to_csv(metrics_csv_path, index=False)
    
    return summary_df, pred_df

def run_all():
    print("=" * 80)
    print("STARTING FULL TUNED MULTI-SEED (N=5) REPRODUCTION FOR BOTH EQUITIES")
    print("=" * 80)
    
    fsr_summary, _ = run_evaluation_for_company("FirstRand", epochs_bnn=200, epochs_lstm=100, s_samples=100, n_restarts_gpr=2)
    dsy_summary, _ = run_evaluation_for_company("Discovery", epochs_bnn=200, epochs_lstm=100, s_samples=100, n_restarts_gpr=2)
    
    combined = pd.DataFrame([
        {"Equity": "FirstRand", **row} for row in fsr_summary.to_dict(orient="records")
    ] + [
        {"Equity": "Discovery", **row} for row in dsy_summary.to_dict(orient="records")
    ])
    
    comb_csv = os.path.join(OUTPUT_DIR, "Trained_Models_Comprehensive_Summary.csv")
    combined.to_csv(comb_csv, index=False)
    
    metrics_json_path = os.path.join(OUTPUT_DIR, "model_evaluation_metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(combined.to_dict(orient="records"), f, indent=2)
    
    print("\n" + "=" * 80)
    print("ALL TUNED MODELS SUCCESSFULLY EVALUATED AND SAVED")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--company", type=str, default="all", choices=["FirstRand", "Discovery", "all"])
    parser.add_argument("--all", action="store_true", help="Run on both FirstRand and Discovery")
    parser.add_argument("--epochs_bnn", type=int, default=200)
    parser.add_argument("--epochs_lstm", type=int, default=100)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--n_restarts_gpr", type=int, default=2)
    args = parser.parse_args()
    
    if args.all or args.company == "all":
        run_all()
    else:
        run_evaluation_for_company(args.company, args.epochs_bnn, args.epochs_lstm, args.samples, args.n_restarts_gpr)
