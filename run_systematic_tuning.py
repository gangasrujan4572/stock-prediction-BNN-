"""
Systematic Hyperparameter Tuning and Validation Benchmark
Logs all experiments to outputs/tuning_log.csv.
All parameter selections and calibrations are performed STRICTLY on the validation split.
"""

import os
import sys
import time
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, DotProduct

sys.path.insert(0, os.path.abspath("."))
import data_processing as dp
import models
import metrics

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
TUNING_LOG_PATH = os.path.join(OUTPUT_DIR, "tuning_log.csv")

def get_prepared_data(stock_name, csv_path):
    raw_df, clean_df = dp.clean_and_prepare_dataset(csv_path)
    splits = dp.split_and_scale_data(clean_df)
    scaler = splits["scaler_info"]
    
    n_train = splits["n_train"] # 3100
    n_val = splits["n_val"]     # 1033
    window_size = 60
    
    all_scaled = (clean_df[dp.MODEL_FEATURES] - scaler["means"]) / scaler["stds"]
    all_scaled_vals = all_scaled.values
    all_close_raw = clean_df["Close"].values
    
    X_train_3d, _ = dp.create_rolling_sequences(
        all_scaled_vals[:n_train],
        all_scaled["Close"].values[:n_train],
        window_size=window_size
    )
    X_val_3d, _ = dp.create_rolling_sequences(
        all_scaled_vals[n_train - window_size : n_train + n_val],
        all_scaled["Close"].values[n_train - window_size : n_train + n_val],
        window_size=window_size
    )
    
    val_actual = all_close_raw[n_train : n_train + n_val]
    train_last_close = all_close_raw[window_size - 1 : n_train - 1]
    train_next_close = all_close_raw[window_size : n_train]
    val_last_close = all_close_raw[n_train - 1 : n_train + n_val - 1]
    
    train_delta = train_next_close - train_last_close
    val_delta = val_actual - val_last_close
    delta_std = np.std(train_delta)
    
    y_train_delta_scaled = (train_delta / delta_std).astype(np.float32)
    y_val_delta_scaled = (val_delta / delta_std).astype(np.float32)
    
    X_train_flat = X_train_3d.reshape(X_train_3d.shape[0], -1)
    X_val_flat = X_val_3d.reshape(X_val_3d.shape[0], -1)
    
    return {
        "stock_name": stock_name,
        "clean_df": clean_df,
        "scaler": scaler,
        "n_train": n_train,
        "n_val": n_val,
        "val_actual": val_actual,
        "val_last_close": val_last_close,
        "delta_std": delta_std,
        "X_train_3d": X_train_3d,
        "y_train_delta": y_train_delta_scaled,
        "X_val_3d": X_val_3d,
        "y_val_delta": y_val_delta_scaled,
        "X_train_flat": X_train_flat,
        "X_val_flat": X_val_flat,
    }

def calibrate_val_interval(val_actual, val_pred_mean, val_pred_std, nominal_picp=0.95):
    """
    Calibrates a variance scaling multiplier or residual noise term strictly on the validation set.
    Finds scale 'c' such that PICP on validation set == nominal_picp (0.95).
    """
    best_c = 1.0
    best_diff = 1.0
    # Search c in [0.5, 10.0]
    for c in np.linspace(0.5, 8.0, 151):
        lower = val_pred_mean - 1.96 * (val_pred_std * c)
        upper = val_pred_mean + 1.96 * (val_pred_std * c)
        picp = np.mean((val_actual >= lower) & (val_actual <= upper))
        if abs(picp - nominal_picp) < best_diff:
            best_diff = abs(picp - nominal_picp)
            best_c = c
            
    cal_lower = val_pred_mean - 1.96 * (val_pred_std * best_c)
    cal_upper = val_pred_mean + 1.96 * (val_pred_std * best_c)
    uq = metrics.compute_uncertainty_metrics(val_actual, cal_lower, cal_upper)
    return best_c, uq

def tune_all():
    tuning_records = []
    print("=" * 80)
    print("STARTING SYSTEMATIC HYPERPARAMETER TUNING ON VALIDATION SET")
    print("=" * 80)
    
    for stock_name, csv_path in [("FirstRand", "FirstRand.csv"), ("Discovery", "Discovery.csv")]:
        data = get_prepared_data(stock_name, csv_path)
        val_actual = data["val_actual"]
        val_last_close = data["val_last_close"]
        delta_std = data["delta_std"]
        
        # -------------------------------------------------------------
        # 1. Tuning GPR
        # -------------------------------------------------------------
        print(f"\n[{stock_name}] Tuning GPR on Validation Split...")
        kernels = [
            ("RBF(10)+White(1e-3)", RBF(10.0, (1e-1, 1e2)) + WhiteKernel(1e-3, (1e-5, 1e1))),
            ("RBF(50)+White(1e-2)", RBF(50.0, (1.0, 1e3)) + WhiteKernel(1e-2, (1e-4, 1.0))),
            ("Matern52(10)+White(1e-3)", Matern(length_scale=10.0, nu=2.5) + WhiteKernel(1e-3, (1e-5, 1e1))),
            ("RBF+Dot+White", RBF(10.0) + DotProduct(1.0) + WhiteKernel(1e-3, (1e-5, 1e1)))
        ]
        
        for k_name, kernel in kernels:
            t0 = time.time()
            try:
                gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, random_state=42)
                gpr.fit(data["X_train_flat"], data["y_train_delta"])
                pred_z, std_z = gpr.predict(data["X_val_flat"], return_std=True)
                val_pred = val_last_close + pred_z * delta_std
                val_std = std_z * delta_std
                
                pt_m = metrics.compute_point_metrics(val_actual, val_pred)
                c_cal, uq_m = calibrate_val_interval(val_actual, val_pred, val_std)
                
                rec = {
                    "stock": stock_name, "model": "GPR", "trial_name": k_name,
                    "lr": np.nan, "batch_size": np.nan, "epochs": np.nan,
                    "kl_weight": np.nan, "dropout": np.nan, "prior_std": np.nan,
                    "hidden_units": str(k_name), "val_mae": pt_m["MAE"],
                    "val_rmse": pt_m["RMSE"], "val_picp": uq_m["PICP"],
                    "val_mpiw": uq_m["MPIW"], "cal_c": c_cal, "time_sec": round(time.time() - t0, 1)
                }
                tuning_records.append(rec)
                print(f"  GPR {k_name:25s} | MAE: {pt_m['MAE']:.2f} | RMSE: {pt_m['RMSE']:.2f} | PICP: {uq_m['PICP']:.3f} | MPIW: {uq_m['MPIW']:.1f}")
            except Exception as e:
                print(f"  GPR {k_name} failed: {e}")

        # -------------------------------------------------------------
        # 2. Tuning Bayesian LSTM & Standard LSTM
        # -------------------------------------------------------------
        print(f"\n[{stock_name}] Tuning Bayesian & Standard LSTM on Validation Split...")
        lstm_grid = [
            {"lr": 0.001, "batch_size": 32, "hidden": 50, "dropout": 0.2, "epochs": 60},
            {"lr": 0.0005, "batch_size": 32, "hidden": 50, "dropout": 0.2, "epochs": 60},
            {"lr": 0.002, "batch_size": 32, "hidden": 50, "dropout": 0.2, "epochs": 60},
            {"lr": 0.001, "batch_size": 64, "hidden": 50, "dropout": 0.2, "epochs": 60},
            {"lr": 0.001, "batch_size": 32, "hidden": 64, "dropout": 0.1, "epochs": 60},
            {"lr": 0.001, "batch_size": 32, "hidden": 32, "dropout": 0.2, "epochs": 60},
            {"lr": 0.001, "batch_size": 32, "hidden": 50, "dropout": 0.1, "epochs": 60},
        ]
        
        train_loader_map = {}
        for cfg in lstm_grid:
            bs = cfg["batch_size"]
            if bs not in train_loader_map:
                td = TensorDataset(torch.tensor(data["X_train_3d"]), torch.tensor(data["y_train_delta"]))
                train_loader_map[bs] = DataLoader(td, batch_size=bs, shuffle=True)
                
        for i, cfg in enumerate(lstm_grid):
            t0 = time.time()
            torch.manual_seed(42 + i)
            # Bayesian LSTM
            blstm = models.BayesianLSTM(input_size=11, hidden_units=cfg["hidden"], dropout_rate=cfg["dropout"])
            optimizer = torch.optim.Adam(blstm.parameters(), lr=cfg["lr"])
            loader = train_loader_map[cfg["batch_size"]]
            
            best_val_loss = float("inf")
            patience = 12
            patience_counter = 0
            best_state = None
            
            for epoch in range(1, cfg["epochs"] + 1):
                blstm.train()
                for bx, by in loader:
                    optimizer.zero_grad()
                    pred = blstm(bx, force_dropout=True).squeeze(-1)
                    loss = nn.functional.mse_loss(pred, by)
                    loss.backward()
                    optimizer.step()
                
                # Validation check for early stopping
                blstm.eval()
                with torch.no_grad():
                    val_out = blstm(torch.tensor(data["X_val_3d"]), force_dropout=False).squeeze(-1)
                    v_loss = nn.functional.mse_loss(val_out, torch.tensor(data["y_val_delta"])).item()
                    if v_loss < best_val_loss:
                        best_val_loss = v_loss
                        best_state = {k: v.cpu() for k, v in blstm.state_dict().items()}
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            break
            
            if best_state is not None:
                blstm.load_state_dict(best_state)
                
            pred_mean_z, pred_std_z = blstm.predict_monte_carlo(data["X_val_3d"], s_samples=100)
            val_pred = val_last_close + pred_mean_z * delta_std
            val_std = pred_std_z * delta_std
            
            pt_m = metrics.compute_point_metrics(val_actual, val_pred)
            c_cal, uq_m = calibrate_val_interval(val_actual, val_pred, val_std)
            
            trial_name = f"BLSTM_h{cfg['hidden']}_lr{cfg['lr']}_bs{cfg['batch_size']}_dr{cfg['dropout']}"
            rec = {
                "stock": stock_name, "model": "Bayesian LSTM", "trial_name": trial_name,
                "lr": cfg["lr"], "batch_size": cfg["batch_size"], "epochs": cfg["epochs"],
                "kl_weight": np.nan, "dropout": cfg["dropout"], "prior_std": np.nan,
                "hidden_units": str(cfg["hidden"]), "val_mae": pt_m["MAE"],
                "val_rmse": pt_m["RMSE"], "val_picp": uq_m["PICP"],
                "val_mpiw": uq_m["MPIW"], "cal_c": c_cal, "time_sec": round(time.time() - t0, 1)
            }
            tuning_records.append(rec)
            print(f"  BLSTM trial {i+1}: {trial_name:35s} | MAE: {pt_m['MAE']:.2f} | RMSE: {pt_m['RMSE']:.2f} | PICP: {uq_m['PICP']:.3f} | MPIW: {uq_m['MPIW']:.1f}")

        # -------------------------------------------------------------
        # 3. Tuning BNN
        # -------------------------------------------------------------
        print(f"\n[{stock_name}] Tuning BNN on Validation Split...")
        bnn_grid = [
            {"lr": 0.001, "batch_size": 32, "kl_weight": 0.001, "dropout": 0.1, "prior_std": 1.0, "epochs": 80},
            {"lr": 0.0005, "batch_size": 32, "kl_weight": 0.001, "dropout": 0.1, "prior_std": 1.0, "epochs": 80},
            {"lr": 0.002, "batch_size": 32, "kl_weight": 0.001, "dropout": 0.1, "prior_std": 1.0, "epochs": 80},
            {"lr": 0.001, "batch_size": 32, "kl_weight": 0.0001, "dropout": 0.1, "prior_std": 1.0, "epochs": 80},
            {"lr": 0.001, "batch_size": 32, "kl_weight": 0.005, "dropout": 0.1, "prior_std": 1.0, "epochs": 80},
            {"lr": 0.001, "batch_size": 64, "kl_weight": 0.001, "dropout": 0.1, "prior_std": 1.0, "epochs": 80},
            {"lr": 0.001, "batch_size": 32, "kl_weight": 0.001, "dropout": 0.05, "prior_std": 1.0, "epochs": 80},
            {"lr": 0.001, "batch_size": 32, "kl_weight": 0.001, "dropout": 0.2, "prior_std": 1.0, "epochs": 80},
            {"lr": 0.001, "batch_size": 32, "kl_weight": 0.001, "dropout": 0.1, "prior_std": 0.5, "epochs": 80},
        ]
        
        flat_loader_map = {}
        for cfg in bnn_grid:
            bs = cfg["batch_size"]
            if bs not in flat_loader_map:
                td = TensorDataset(torch.tensor(data["X_train_flat"]), torch.tensor(data["y_train_delta"]))
                flat_loader_map[bs] = DataLoader(td, batch_size=bs, shuffle=True)

        for i, cfg in enumerate(bnn_grid):
            t0 = time.time()
            torch.manual_seed(42 + i)
            bnn = models.BayesianNeuralNetwork(
                in_features=660,
                dropout_rate=cfg["dropout"],
                prior_std=cfg["prior_std"]
            )
            optimizer = torch.optim.Adam(bnn.parameters(), lr=cfg["lr"])
            loader = flat_loader_map[cfg["batch_size"]]
            
            best_val_loss = float("inf")
            patience = 15
            patience_counter = 0
            best_state = None
            
            for epoch in range(1, cfg["epochs"] + 1):
                bnn.train()
                for bx, by in loader:
                    optimizer.zero_grad()
                    pred = bnn(bx, sample=True).squeeze(-1)
                    mse_loss = nn.functional.mse_loss(pred, by)
                    kl_loss = bnn.total_kl() / len(data["X_train_flat"])
                    loss = mse_loss + cfg["kl_weight"] * kl_loss
                    loss.backward()
                    optimizer.step()
                
                # Val check
                bnn.eval()
                with torch.no_grad():
                    val_out = bnn(torch.tensor(data["X_val_flat"]), sample=False).squeeze(-1)
                    v_loss = nn.functional.mse_loss(val_out, torch.tensor(data["y_val_delta"])).item()
                    if v_loss < best_val_loss:
                        best_val_loss = v_loss
                        best_state = {k: v.cpu() for k, v in bnn.state_dict().items()}
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            break
                            
            if best_state is not None:
                bnn.load_state_dict(best_state)
                
            pred_mean_z, pred_std_z = bnn.predict_monte_carlo(data["X_val_flat"], s_samples=100)
            val_pred = val_last_close + pred_mean_z * delta_std
            val_std = pred_std_z * delta_std
            
            pt_m = metrics.compute_point_metrics(val_actual, val_pred)
            c_cal, uq_m = calibrate_val_interval(val_actual, val_pred, val_std)
            
            trial_name = f"BNN_lr{cfg['lr']}_kl{cfg['kl_weight']}_dr{cfg['dropout']}_pstd{cfg['prior_std']}"
            rec = {
                "stock": stock_name, "model": "BNN", "trial_name": trial_name,
                "lr": cfg["lr"], "batch_size": cfg["batch_size"], "epochs": cfg["epochs"],
                "kl_weight": cfg["kl_weight"], "dropout": cfg["dropout"],
                "prior_std": cfg["prior_std"], "hidden_units": "64-32",
                "val_mae": pt_m["MAE"], "val_rmse": pt_m["RMSE"],
                "val_picp": uq_m["PICP"], "val_mpiw": uq_m["MPIW"],
                "cal_c": c_cal, "time_sec": round(time.time() - t0, 1)
            }
            tuning_records.append(rec)
            print(f"  BNN trial {i+1}: {trial_name:40s} | MAE: {pt_m['MAE']:.2f} | RMSE: {pt_m['RMSE']:.2f} | PICP: {uq_m['PICP']:.3f} | MPIW: {uq_m['MPIW']:.1f}")

    df_tuning = pd.DataFrame(tuning_records)
    df_tuning.to_csv(TUNING_LOG_PATH, index=False)
    print(f"\n[DONE] Saved tuning log to {TUNING_LOG_PATH} ({len(df_tuning)} trials).")
    
    # Print summary of best configurations per stock and model
    print("\n" + "=" * 80)
    print("BEST CONFIGURATIONS ON VALIDATION SET (MIN VAL MAE)")
    print("=" * 80)
    for stock in ["FirstRand", "Discovery"]:
        print(f"\n--- {stock} Top Validation Configurations ---")
        df_stock = df_tuning[df_tuning["stock"] == stock]
        for model in ["GPR", "Bayesian LSTM", "BNN"]:
            df_m = df_stock[df_stock["model"] == model]
            if not df_m.empty:
                best_row = df_m.sort_values(by="val_mae").iloc[0]
                print(f"  {model:15s} | Trial: {best_row['trial_name']} | Val MAE: {best_row['val_mae']:.2f} | Val RMSE: {best_row['val_rmse']:.2f} | PICP: {best_row['val_picp']:.3f} | Cal C: {best_row['cal_c']:.2f}")

if __name__ == "__main__":
    tune_all()
