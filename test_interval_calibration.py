import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath("."))
import data_processing as dp
import models
import metrics

def test_interval_calibration():
    print("=======================================================")
    print("TESTING INTERVAL CALIBRATION METHODS ON VALIDATION SET")
    print("=======================================================")
    
    for stock_name, csv_path in [("FirstRand", "FirstRand.csv"), ("Discovery", "Discovery.csv")]:
        raw_df, clean_df = dp.clean_and_prepare_dataset(csv_path)
        splits = dp.split_and_scale_data(clean_df)
        scaler = splits["scaler_info"]
        n_train = splits["n_train"]
        n_val = splits["n_val"]
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
        
        # Train BNN
        torch.manual_seed(42)
        bnn = models.BayesianNeuralNetwork(in_features=660, dropout_rate=0.1, prior_std=1.0)
        optimizer = torch.optim.Adam(bnn.parameters(), lr=0.001)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.tensor(X_train_flat), torch.tensor(y_train_delta_scaled)),
            batch_size=32, shuffle=True
        )
        for epoch in range(40):
            for bx, by in loader:
                optimizer.zero_grad()
                pred = bnn(bx, sample=True).squeeze(-1)
                loss = torch.nn.functional.mse_loss(pred, by) + 0.001 * bnn.total_kl() / len(X_train_flat)
                loss.backward()
                optimizer.step()
                
        pred_mean_z, pred_std_z = bnn.predict_monte_carlo(X_val_flat, s_samples=100)
        val_pred = val_last_close + pred_mean_z * delta_std
        val_std_epistemic = pred_std_z * delta_std
        
        # 1. Raw epistemic (paper's definition)
        raw_lower = val_pred - 1.96 * val_std_epistemic
        raw_upper = val_pred + 1.96 * val_std_epistemic
        uq_raw = metrics.compute_uncertainty_metrics(val_actual, raw_lower, raw_upper)
        print(f"\n[{stock_name}] BNN Validation Uncertainty:")
        print(f"  1. Raw Epistemic (Paper def): PICP = {uq_raw['PICP']:.4f}, MPIW = {uq_raw['MPIW']:.2f}")
        
        # 2. Total variance with validation residual noise: sigma_total^2 = sigma_epistemic^2 + sigma_res_val^2
        # where sigma_res_val is computed from validation residuals
        val_residuals = val_actual - val_pred
        sigma_res_val = np.std(val_residuals)
        sigma_total = np.sqrt(val_std_epistemic**2 + sigma_res_val**2)
        res_lower = val_pred - 1.96 * sigma_total
        res_upper = val_pred + 1.96 * sigma_total
        uq_res = metrics.compute_uncertainty_metrics(val_actual, res_lower, res_upper)
        print(f"  2. Total Variance (Epistemic + Val Residual Noise): PICP = {uq_res['PICP']:.4f}, MPIW = {uq_res['MPIW']:.2f}")

        # 3. Validation Calibrated Scaling Factor c
        # Search c to get PICP closest to 0.95
        best_c = 1.0
        best_diff = 1.0
        for c in np.linspace(0.5, 10.0, 201):
            l = val_pred - 1.96 * (val_std_epistemic * c)
            u = val_pred + 1.96 * (val_std_epistemic * c)
            p = np.mean((val_actual >= l) & (val_actual <= u))
            if abs(p - 0.95) < best_diff:
                best_diff = abs(p - 0.95)
                best_c = c
        c_lower = val_pred - 1.96 * (val_std_epistemic * best_c)
        c_upper = val_pred + 1.96 * (val_std_epistemic * best_c)
        uq_c = metrics.compute_uncertainty_metrics(val_actual, c_lower, c_upper)
        print(f"  3. Calibrated Scaling Factor (c = {best_c:.2f}): PICP = {uq_c['PICP']:.4f}, MPIW = {uq_c['MPIW']:.2f}")

if __name__ == "__main__":
    test_interval_calibration()
