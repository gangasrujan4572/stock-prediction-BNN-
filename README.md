# Bayesian Deep Learning and Probabilistic Forecasting of Stock Prices

This repository provides an honest, empirical reproduction of the methodology, model architectures, and probabilistic forecasting pipeline described in:

> **Nelufhangani, N., & Maposa, D. (2026).** *Bayesian Deep Learning and Probabilistic Forecasting of Stock Prices*. **Algorithms**, 19(5), 391. [https://doi.org/10.3390/a19050391](https://doi.org/10.3390/a19050391)

---

## ⚡ Quick Start: Running the Reproduction Pipeline

All required dependencies (`torch`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `statsmodels`, `tabulate`, `pypdf`, `yfinance`) are installed in the local environment.

### 1. Train and Evaluate Models from Scratch (10-Seed Tuned Pipeline)

To train all 5 models (BNN, Bayesian LSTM, Standard LSTM, GPR, ARIMA) on both **FirstRand** and **Discovery** across 10 independent random seeds:

```powershell
python train_and_evaluate.py --all
```

Or for an individual equity:
```powershell
python train_and_evaluate.py --company FirstRand
python train_and_evaluate.py --company Discovery
```

This pipeline executes:
- Multi-seed training (10 seeds: `[42, 43, 44, 45, 46, 47, 48, 49, 50, 51]`)
- Stationary 1-step relative delta price targeting with causal price reconstruction
- Validation-calibrated uncertainty estimation for ~0.95 PICP coverage
- Saves predictions and metrics to `outputs/`

### 2. Generate All Paper Tables and Figures

Once models are evaluated, generate all 12 tables, 10 figures, and the 3-way comparison report strictly from the saved outputs:

```powershell
python generate_paper_results.py
```

This command produces:
1. **Tables 2 & 3**: Descriptive statistics for FirstRand and Discovery.
2. **Figures 1–4**: Historical closing prices and 13-feature correlation matrix heatmaps.
3. **Tables 4, 6, 8**: Model test performance metrics (MAE, RMSE, MSE) computed from test predictions.
4. **Tables 5, 7, 9**: 10-day sample actual vs. predicted prices with genuine 95% confidence intervals.
5. **Table 10**: Comparative performance across all Bayesian models.
6. **Table 11**: Uncertainty quantification (PICP coverage and MPIW sharpness at 95% level).
7. **Table 12**: Comparison with non-Bayesian baselines (ARIMA and Standard LSTM).
8. **Figures 5–10**: Forecast trajectories with 95% confidence intervals ($\mu \pm 1.96\sigma$).
9. **`reproduction_vs_paper_comparison.csv`**: Automated side-by-side comparison with the paper's reported values.
10. **`baseline_vs_tuned_vs_paper_comparison.csv`**: Three-way comparison between baseline run, tuned run, and published paper values.

### 3. Verify Reproduction Integrity

To verify that no hardcoded metric tables exist and that all numbers match raw model predictions recomputed from scratch:

```powershell
python verify_reproduction.py
```

---

## 📊 Three-Way Progression: Baseline Run vs. Tuned Run vs. Paper

| Company | Model | Metric | Baseline Run (Un-tuned) | Tuned Run (10-Seed Mean) | Paper Target |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **FirstRand** | **GPR** | MAE | 182.40 | **81.87** | 72.27 |
| **FirstRand** | **GPR** | RMSE | 240.43 | **110.56** | 102.80 |
| **FirstRand** | **GPR** | PICP | 0.58 | **0.97** | 0.94 |
| **FirstRand** | **Bayesian LSTM** | MAE | 945.03 | **83.07** | 133.49 |
| **FirstRand** | **Bayesian LSTM** | RMSE | 1,131.50 | **111.67** | 175.21 |
| **FirstRand** | **Bayesian LSTM** | PICP | 0.20 | **0.96** | 0.96 |
| **FirstRand** | **BNN** | MAE | 351.76 | **82.36** | 63.78 |
| **FirstRand** | **BNN** | RMSE | 380.33 | **111.08** | 83.29 |
| **FirstRand** | **BNN** | PICP | 0.76 | **0.96** | 0.95 |
| **FirstRand** | **Standard LSTM** | MAE | 769.98 | **82.97** | 89.14 |
| **FirstRand** | **ARIMA** | MAE | 1,585.71 | **81.65** | 198.43 |
| **Discovery** | **GPR** | MAE | 904.21 | **166.42** | 115.15 |
| **Discovery** | **GPR** | RMSE | 1,519.99 | **236.29** | 226.19 |
| **Discovery** | **GPR** | PICP | 0.62 | **0.98** | 0.92 |
| **Discovery** | **Bayesian LSTM** | MAE | 1,090.08 | **176.36** | 337.39 |
| **Discovery** | **Bayesian LSTM** | RMSE | 1,631.11 | **245.95** | 452.78 |
| **Discovery** | **Bayesian LSTM** | PICP | 0.72 | **0.97** | 0.95 |
| **Discovery** | **BNN** | MAE | 956.78 | **175.40** | 144.56 |
| **Discovery** | **BNN** | RMSE | 1,069.61 | **245.91** | 201.50 |
| **Discovery** | **BNN** | PICP | 0.44 | **0.97** | 0.94 |
| **Discovery** | **Standard LSTM** | MAE | 726.91 | **171.66** | 189.73 |
| **Discovery** | **ARIMA** | MAE | 3,000.48 | **166.75** | 412.56 |

---

## 🛠️ Hyperparameters & Methodological Deviations

All changes and tuning configurations are documented in detail below:

1. **Target Formulation & Causal Price Reconstruction (Deviation #1)**:
   - *Issue in Baseline*: The test period (2021–2025) experiences price levels far above the training domain (FirstRand train max 5,767 vs test max 8,679; Discovery train max 15,450 vs test max 21,178). Regressing normalized price levels leads to extrapolation breakdown.
   - *Fix*: Models predict the normalized 1-step price change relative to the last known Close in the window ($\Delta Close_{t+1} / \sigma_{\Delta,\text{train}}$), followed by causal price reconstruction: $\hat{Close}_{t+1} = Close_t + \widehat{\Delta Close}_{t+1}$.
2. **Uncertainty Interval Calibration (Deviation #2)**:
   - *Issue in Baseline*: MC Dropout and variational inference measure only epistemic (parameter) uncertainty ($\sigma_{\text{MC}} \approx 5-10$ price points), ignoring market observation noise / aleatoric variance ($\sigma_{\text{market}} \approx 50-125$).
   - *Fix*: Total predictive variance is evaluated using validation residual noise: $\sigma_{\text{total}} = \sqrt{\sigma_{\text{MC}}^2 + \sigma_{\text{res,val}}^2}$, calibrated strictly on the validation set.
3. **ARIMA 1-Step Rolling Walk-Forward (Deviation #3)**:
   - *Fix*: ARIMA(1, 1, 1) is evaluated using standard 1-step-ahead walk-forward updating across the test set rather than a static 1,034-step forward projection.
4. **Validation Early Stopping & Loss Formulations**:
   - Neural models employ validation loss early stopping (patience 20–25) with Smooth L1 / Huber criterion for recurrent networks and ELBO for variational networks.

---

## 📁 Repository Structure

```text
pr/
│
├── data_processing.py        # Data loading, Wilder RSI, feature engineering, Z-score scaling, windowing
├── models.py                 # Architectures: BNN (BBB), Bayesian LSTM (MC Dropout), GPR, Standard LSTM, ARIMA
├── metrics.py                # Point metrics (MAE, RMSE, MSE) and uncertainty metrics (PICP, MPIW)
├── plots.py                  # Matplotlib visualizer for Figures 1–10
├── generate_paper_results.py # Generates Tables 2–12, Figures 1–10, and comparison tables from real outputs
├── train_and_evaluate.py     # End-to-end 10-seed training and evaluation pipeline
├── verify_reproduction.py    # Automated test suite auditing integrity and recomputing metrics
├── MODEL_REPRODUCTION_MASTER_DOCUMENT.txt # Comprehensive technical & defense documentation
├── paper_settings.md         # Reference specifications: stated vs unstated paper settings
├── paper_reported_results.csv # Published numbers from the original paper
│
├── FirstRand.csv             # Historical daily market data for FirstRand Limited (FSR.JO)
├── Discovery.csv             # Historical daily market data for Discovery Limited (DSY.JO)
├── algorithms-19-00391.pdf   # Original research article (Nelufhangani & Maposa, 2026)
│
└── outputs/                  # Genuine generated figures, tables, .npz predictions, and markdown reports
```
