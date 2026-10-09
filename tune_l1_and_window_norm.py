import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern

sys.path.insert(0, os.path.abspath("."))
import data_processing as dp
import models
import metrics

def test_l1_and_scaling():
    print("=======================================================")
    print("TESTING L1 / HUBER LOSS & FEATURE NORMALIZATION ON VAL SET")
    print("=======================================================")
    
    for stock_name, csv_path in [("FirstRand", "FirstRand.csv"), ("Discovery", "Discovery.csv")]:
        raw_df, clean_df = dp.clean_and_prepare_dataset(csv_path)
        ds = dp.generate_model_datasets(clean_df)
        
        val_actual = ds["val_actual_unscaled"]
        val_last_close = ds["val_last_close"]
        delta_std = ds["delta_std"]
        
        # 1. BNN with MSE Loss vs Huber Loss vs L1 Loss
        print(f"\n[{stock_name}] Testing BNN Loss Formulations on Validation:")
        for loss_type, criterion in [("MSE", nn.MSELoss()), ("Huber (SmoothL1)", nn.SmoothL1Loss(beta=0.5)), ("L1", nn.L1Loss())]:
            torch.manual_seed(42)
            bnn = models.BayesianNeuralNetwork(in_features=660, dropout_rate=0.1, prior_std=1.0)
            optimizer = torch.optim.Adam(bnn.parameters(), lr=0.001)
            loader = DataLoader(TensorDataset(torch.tensor(ds["X_train_flat"]), torch.tensor(ds["y_train_delta"])), batch_size=32, shuffle=True)
            
            best_val = float("inf")
            best_state = None
            for epoch in range(50):
                bnn.train()
                for bx, by in loader:
                    optimizer.zero_grad()
                    pred = bnn(bx, sample=True).squeeze(-1)
                    l = criterion(pred, by) + 0.001 * bnn.total_kl() / len(ds["X_train_flat"])
                    l.backward()
                    optimizer.step()
                bnn.eval()
                with torch.no_grad():
                    vp = bnn(torch.tensor(ds["X_val_flat"]), sample=False).squeeze(-1)
                    vl = criterion(vp, torch.tensor(ds["y_val_delta"])).item()
                    if vl < best_val:
                        best_val = vl
                        best_state = {k: v.cpu() for k, v in bnn.state_dict().items()}
            bnn.load_state_dict(best_state)
            pred_mean_z, _ = bnn.predict_monte_carlo(ds["X_val_flat"], s_samples=50)
            val_pred = val_last_close + pred_mean_z * delta_std
            m = metrics.compute_point_metrics(val_actual, val_pred)
            print(f"  BNN with {loss_type:16s} | Val MAE: {m['MAE']:.2f} | Val RMSE: {m['RMSE']:.2f}", flush=True)

        # 2. Bayesian LSTM with MSE vs Huber vs L1 Loss
        print(f"\n[{stock_name}] Testing Bayesian LSTM Loss Formulations on Validation:", flush=True)
        for loss_type, criterion in [("MSE", nn.MSELoss()), ("Huber (SmoothL1)", nn.SmoothL1Loss(beta=0.5)), ("L1", nn.L1Loss())]:
            torch.manual_seed(42)
            blstm = models.BayesianLSTM(input_size=11, hidden_units=50, dropout_rate=0.2)
            optimizer = torch.optim.Adam(blstm.parameters(), lr=0.001)
            loader = DataLoader(TensorDataset(torch.tensor(ds["X_train_3d"]), torch.tensor(ds["y_train_delta"])), batch_size=32, shuffle=True)
            
            best_val = float("inf")
            best_state = None
            for epoch in range(40):
                blstm.train()
                for bx, by in loader:
                    optimizer.zero_grad()
                    pred = blstm(bx, force_dropout=True).squeeze(-1)
                    l = criterion(pred, by)
                    l.backward()
                    optimizer.step()
                blstm.eval()
                with torch.no_grad():
                    vp = blstm(torch.tensor(ds["X_val_3d"]), force_dropout=False).squeeze(-1)
                    vl = criterion(vp, torch.tensor(ds["y_val_delta"])).item()
                    if vl < best_val:
                        best_val = vl
                        best_state = {k: v.cpu() for k, v in blstm.state_dict().items()}
            blstm.load_state_dict(best_state)
            pred_mean_z, _ = blstm.predict_monte_carlo(ds["X_val_3d"], s_samples=50)
            val_pred = val_last_close + pred_mean_z * delta_std
            m = metrics.compute_point_metrics(val_actual, val_pred)
            print(f"  BLSTM with {loss_type:16s} | Val MAE: {m['MAE']:.2f} | Val RMSE: {m['RMSE']:.2f}", flush=True)

if __name__ == "__main__":
    test_l1_and_scaling()
