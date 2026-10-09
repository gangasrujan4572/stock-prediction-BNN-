import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

sys.path.insert(0, os.path.abspath("."))
import data_processing as dp
import models
import metrics

def run_experiment():
    for stock_name, csv_path in [("FirstRand", "FirstRand.csv"), ("Discovery", "Discovery.csv")]:
        print(f"\n=======================================================", flush=True)
        print(f"EVALUATING TARGET & SCALING FORMULATIONS ON VAL SET: {stock_name}", flush=True)
        print(f"=======================================================", flush=True)
        
        raw_df, clean_df = dp.clean_and_prepare_dataset(csv_path)
        splits = dp.split_and_scale_data(clean_df)
        scaler = splits["scaler_info"]
        
        n_train = splits["n_train"] # 3100
        n_val = splits["n_val"]     # 1033
        window_size = 60
        
        # Standard features (Z-score standardized on train)
        all_scaled = (clean_df[dp.MODEL_FEATURES] - scaler["means"]) / scaler["stds"]
        all_scaled_vals = all_scaled.values
        all_close_raw = clean_df["Close"].values
        
        # Train and Val windows
        X_train_3d, y_train_level = dp.create_rolling_sequences(
            all_scaled_vals[:n_train],
            all_scaled["Close"].values[:n_train],
            window_size=window_size
        )
        X_val_3d, y_val_level = dp.create_rolling_sequences(
            all_scaled_vals[n_train - window_size : n_train + n_val],
            all_scaled["Close"].values[n_train - window_size : n_train + n_val],
            window_size=window_size
        )
        
        val_actual = all_close_raw[n_train : n_train + n_val] # length 1033
        
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
        
        print(f"Train samples: {len(X_train_3d)}, Val samples: {len(X_val_3d)}", flush=True)
        print(f"Train Close Range: [{clean_df['Close'].iloc[:n_train].min():.1f}, {clean_df['Close'].iloc[:n_train].max():.1f}]", flush=True)
        print(f"Val Close Range:   [{val_actual.min():.1f}, {val_actual.max():.1f}]", flush=True)
        print(f"Delta std: {delta_std:.3f}", flush=True)
        
        # 1. Test GPR
        print("\n--- GPR Validation ---", flush=True)
        t0 = time.time()
        # Level
        gpr_a = GaussianProcessRegressor(kernel=RBF(10.0) + WhiteKernel(1e-3), n_restarts_optimizer=0, random_state=42)
        gpr_a.fit(X_train_flat, y_train_level)
        pred_val_a_z, std_val_a_z = gpr_a.predict(X_val_flat, return_std=True)
        pred_val_a = pred_val_a_z * scaler["close_std"] + scaler["close_mean"]
        m_gpr_a = metrics.compute_point_metrics(val_actual, pred_val_a)
        print(f"GPR [Level]: MAE={m_gpr_a['MAE']:.2f}, RMSE={m_gpr_a['RMSE']:.2f} ({time.time()-t0:.1f}s)", flush=True)
        
        # Delta
        t0 = time.time()
        gpr_b = GaussianProcessRegressor(kernel=RBF(10.0) + WhiteKernel(1e-3), n_restarts_optimizer=0, random_state=42)
        gpr_b.fit(X_train_flat, y_train_delta_scaled)
        pred_val_b_d, std_val_b_d = gpr_b.predict(X_val_flat, return_std=True)
        pred_val_b = val_last_close + pred_val_b_d * delta_std
        m_gpr_b = metrics.compute_point_metrics(val_actual, pred_val_b)
        print(f"GPR [Delta]: MAE={m_gpr_b['MAE']:.2f}, RMSE={m_gpr_b['RMSE']:.2f} ({time.time()-t0:.1f}s)", flush=True)
        
        # 2. Test Bayesian LSTM
        print("\n--- Bayesian LSTM Validation ---", flush=True)
        torch.manual_seed(42)
        blstm_a = models.BayesianLSTM(input_size=11, hidden_units=50, dropout_rate=0.2)
        opt_a = torch.optim.Adam(blstm_a.parameters(), lr=0.001)
        loader_a = DataLoader(TensorDataset(torch.tensor(X_train_3d), torch.tensor(y_train_level)), batch_size=32, shuffle=True)
        for epoch in range(40):
            for bx, by in loader_a:
                opt_a.zero_grad()
                loss = nn.functional.mse_loss(blstm_a(bx, force_dropout=True).squeeze(-1), by)
                loss.backward()
                opt_a.step()
        mean_z_a, _ = blstm_a.predict_monte_carlo(X_val_3d, s_samples=50)
        pred_blstm_a = mean_z_a * scaler["close_std"] + scaler["close_mean"]
        m_blstm_a = metrics.compute_point_metrics(val_actual, pred_blstm_a)
        print(f"Bayesian LSTM [Level]: MAE={m_blstm_a['MAE']:.2f}, RMSE={m_blstm_a['RMSE']:.2f}", flush=True)
        
        torch.manual_seed(42)
        blstm_b = models.BayesianLSTM(input_size=11, hidden_units=50, dropout_rate=0.2)
        opt_b = torch.optim.Adam(blstm_b.parameters(), lr=0.001)
        loader_b = DataLoader(TensorDataset(torch.tensor(X_train_3d), torch.tensor(y_train_delta_scaled)), batch_size=32, shuffle=True)
        for epoch in range(40):
            for bx, by in loader_b:
                opt_b.zero_grad()
                loss = nn.functional.mse_loss(blstm_b(bx, force_dropout=True).squeeze(-1), by)
                loss.backward()
                opt_b.step()
        mean_d_b, _ = blstm_b.predict_monte_carlo(X_val_3d, s_samples=50)
        pred_blstm_b = val_last_close + mean_d_b * delta_std
        m_blstm_b = metrics.compute_point_metrics(val_actual, pred_blstm_b)
        print(f"Bayesian LSTM [Delta]: MAE={m_blstm_b['MAE']:.2f}, RMSE={m_blstm_b['RMSE']:.2f}", flush=True)
        
        # 3. Test BNN
        print("\n--- BNN Validation ---", flush=True)
        torch.manual_seed(42)
        bnn_a = models.BayesianNeuralNetwork(in_features=660, dropout_rate=0.1)
        opt_bnn_a = torch.optim.Adam(bnn_a.parameters(), lr=0.001)
        loader_bnn_a = DataLoader(TensorDataset(torch.tensor(X_train_flat), torch.tensor(y_train_level)), batch_size=32, shuffle=True)
        for epoch in range(40):
            for bx, by in loader_bnn_a:
                opt_bnn_a.zero_grad()
                pred = bnn_a(bx, sample=True).squeeze(-1)
                loss = nn.functional.mse_loss(pred, by) + 0.001 * bnn_a.total_kl() / len(X_train_flat)
                loss.backward()
                opt_bnn_a.step()
        mean_z_bnn_a, _ = bnn_a.predict_monte_carlo(X_val_flat, s_samples=50)
        pred_bnn_a = mean_z_bnn_a * scaler["close_std"] + scaler["close_mean"]
        m_bnn_a = metrics.compute_point_metrics(val_actual, pred_bnn_a)
        print(f"BNN [Level]: MAE={m_bnn_a['MAE']:.2f}, RMSE={m_bnn_a['RMSE']:.2f}", flush=True)
        
        torch.manual_seed(42)
        bnn_b = models.BayesianNeuralNetwork(in_features=660, dropout_rate=0.1)
        opt_bnn_b = torch.optim.Adam(bnn_b.parameters(), lr=0.001)
        loader_bnn_b = DataLoader(TensorDataset(torch.tensor(X_train_flat), torch.tensor(y_train_delta_scaled)), batch_size=32, shuffle=True)
        for epoch in range(40):
            for bx, by in loader_bnn_b:
                opt_bnn_b.zero_grad()
                pred = bnn_b(bx, sample=True).squeeze(-1)
                loss = nn.functional.mse_loss(pred, by) + 0.001 * bnn_b.total_kl() / len(X_train_flat)
                loss.backward()
                opt_bnn_b.step()
        mean_d_bnn_b, _ = bnn_b.predict_monte_carlo(X_val_flat, s_samples=50)
        pred_bnn_b = val_last_close + mean_d_bnn_b * delta_std
        m_bnn_b = metrics.compute_point_metrics(val_actual, pred_bnn_b)
        print(f"BNN [Delta]: MAE={m_bnn_b['MAE']:.2f}, RMSE={m_bnn_b['RMSE']:.2f}", flush=True)

if __name__ == "__main__":
    run_experiment()
