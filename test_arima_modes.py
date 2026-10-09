import os
import sys
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

sys.path.insert(0, os.path.abspath("."))
import data_processing as dp
import metrics

def test_arima_modes():
    print("=======================================================")
    print("TESTING ARIMA ROLLING 1-STEP VS MULTI-STEP ON VAL SET")
    print("=======================================================")
    
    for stock_name, csv_path in [("FirstRand", "FirstRand.csv"), ("Discovery", "Discovery.csv")]:
        raw_df, clean_df = dp.clean_and_prepare_dataset(csv_path)
        splits = dp.split_and_scale_data(clean_df)
        train_series = splits["train_df"]["Close"].values
        val_series = splits["val_df"]["Close"].values
        n_train = len(train_series)
        n_val = len(val_series)
        
        # 1. Multi-step static forecast (old baseline)
        model_static = ARIMA(train_series, order=(1, 1, 1)).fit()
        static_preds = model_static.forecast(steps=n_val)
        m_static = metrics.compute_point_metrics(val_series, static_preds)
        print(f"\n{stock_name} ARIMA Multi-step static (1033 steps):")
        print(f"  MAE: {m_static['MAE']:.2f}, RMSE: {m_static['RMSE']:.2f}")
        
        # 2. Rolling 1-step-ahead walk-forward forecast
        # Fit model on training, then iteratively append new observation without full re-estimation
        history = list(train_series)
        rolling_preds = []
        
        # Using apply / append for fast 1-step updates
        res = ARIMA(train_series, order=(1, 1, 1)).fit()
        # Alternatively compute 1-step predictions over validation
        full_series = np.concatenate([train_series, val_series])
        res_full = ARIMA(full_series, order=(1, 1, 1)).filter(res.params)
        # 1-step ahead forecasts for validation indices
        rolling_preds = res_full.predict(start=n_train, end=n_train + n_val - 1)
        m_rolling = metrics.compute_point_metrics(val_series, rolling_preds)
        print(f"{stock_name} ARIMA Rolling 1-step-ahead (1,1,1):")
        print(f"  MAE: {m_rolling['MAE']:.2f}, RMSE: {m_rolling['RMSE']:.2f}")

        # Try ARIMA(2, 1, 2) or ARIMA(0, 1, 0) (random walk)
        for order in [(0, 1, 0), (1, 1, 0), (0, 1, 1), (2, 1, 2)]:
            try:
                res_ord = ARIMA(train_series, order=order).fit()
                res_full_ord = ARIMA(full_series, order=order).filter(res_ord.params)
                preds_ord = res_full_ord.predict(start=n_train, end=n_train + n_val - 1)
                m_ord = metrics.compute_point_metrics(val_series, preds_ord)
                print(f"  Order {order}: MAE: {m_ord['MAE']:.2f}, RMSE: {m_ord['RMSE']:.2f}")
            except Exception as e:
                print(f"  Order {order} error: {e}")

if __name__ == "__main__":
    test_arima_modes()
