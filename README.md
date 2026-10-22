# Bayesian Deep Learning and Probabilistic Forecasting of Stock Prices

This repository provides an honest, empirical reproduction of the methodology, model architectures, and probabilistic forecasting pipeline described in:

> **Nelufhangani, N., & Maposa, D. (2026).** *Bayesian Deep Learning and Probabilistic Forecasting of Stock Prices*. **Algorithms**, 19(5), 391. [https://doi.org/10.3390/a19050391](https://doi.org/10.3390/a19050391)

---

## ⚡ Quick Start: Running the Reproduction Pipeline

All required dependencies (`torch`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `statsmodels`, `tabulate`, `pypdf`, `python-pptx`, `nbformat`, `nbconvert`) are installed in the environment.

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
- Validation-calibrated uncertainty estimation for ~0.95–0.97 nominal PICP coverage
- Saves raw prediction tensors, evaluation metrics, and logs to [`outputs/`](file:///c:/Users/rishi/OneDrive/Desktop/pr%20final/outputs)

---

### 2. Generate All Paper Tables and Figures

Once models are evaluated, dynamically generate all 12 tables, 10 figures, and the 3-way comparison report strictly from the saved outputs:

```powershell
python generate_paper_results.py
```

This command produces:
1. **Tables 2 & 3**: Descriptive statistics for FirstRand and Discovery.
2. **Figures 1–4**: Historical closing prices and 13-feature correlation matrix heatmaps.
3. **Tables 4, 6, 8**: Model test performance metrics (MAE, RMSE, MSE) computed dynamically from test predictions.
4. **Tables 5, 7, 9**: 10-day sample actual vs. predicted prices with genuine 95% confidence intervals.
5. **Table 10**: Comparative performance across all Bayesian models.
6. **Table 11**: Uncertainty quantification (PICP coverage and MPIW sharpness at 95% nominal level).
7. **Table 12**: Comparison with non-Bayesian baselines (ARIMA and Standard LSTM).
8. **Figures 5–10**: Forecast trajectories with 95% confidence intervals ($\mu \pm 1.96\sigma$).
9. **`reproduction_vs_paper_comparison.csv`**: Automated side-by-side comparison with the paper's reported values.
10. **`baseline_vs_tuned_vs_paper_comparison.csv`**: Three-way comparison between baseline run, tuned run, and published paper values.

---

### 3. Verify Reproduction Integrity

To verify that zero hardcoded metric tables exist, no synthetic noise is injected, and that all numbers match raw model predictions recomputed from scratch:

```powershell
python verify_reproduction.py
```

---

### 4. Generate Research Presentation (20-Slide Defense Deck)

To build the executive, high-resolution 20-slide research defense presentation in 16:9 widescreen format:

```powershell
python generate_20_slides_presentation.py
```

This generates [`Bayesian_Stock_Prediction_20_Slides.pptx`](file:///c:/Users/rishi/OneDrive/Desktop/pr%20final/Bayesian_Stock_Prediction_20_Slides.pptx), which includes:
- Clear mathematical formulas (Bayes by Backprop, MC Dropout, GPR Marginal Likelihood, PICP, MPIW)
- Exact empirical comparison tables from the 10-seed experiment
- High-resolution embedded figures (Figures 1–10 and All-Model comparative charts)
- Defense arguments and practical deployment recommendations

---

### 5. Interactive Jupyter Notebook

The repository includes a self-contained, fully executed research notebook:
- **Notebook File**: [`Bayesian_Stock_Prediction_Notebook.ipynb`](file:///c:/Users/rishi/OneDrive/Desktop/pr%20final/Bayesian_Stock_Prediction_Notebook.ipynb)
- **Builder Script**: To rebuild and re-execute the notebook from scratch:
  ```powershell
  python build_full_notebook.py
  ```

---

## 📊 Three-Way Progression: Baseline Run vs. Tuned Run vs. Paper

The empirical results across 10 random seeds demonstrate how resolving out-of-distribution non-stationarity aligns empirical results with the paper's published findings:

| Company | Model | Metric | Baseline Run (Un-tuned Level) | Tuned Run (10-Seed Mean) | Paper Target | Status & Observation |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **FirstRand** | **GPR** | MAE | 182.40 | **81.87** | 72.27 | -55% error reduction $\to$ close to paper |
| **FirstRand** | **GPR** | RMSE | 240.43 | **110.56** | 102.80 | -54% error reduction |
| **FirstRand** | **GPR** | PICP | 0.58 | **0.97** | 0.94 | Calibrated nominal 95% coverage |
| **FirstRand** | **Bayesian LSTM** | MAE | 945.03 | **83.45** | 133.49 | -91% error reduction (outperforms paper) |
| **FirstRand** | **Bayesian LSTM** | RMSE | 1,131.50 | **112.12** | 175.21 | -90% error reduction |
| **FirstRand** | **Bayesian LSTM** | PICP | 0.20 | **0.96** | 0.96 | Exact nominal coverage match |
| **FirstRand** | **BNN** | MAE | 351.76 | **81.77** | 63.78 | -76% error reduction |
| **FirstRand** | **BNN** | RMSE | 380.33 | **110.49** | 83.29 | -71% error reduction |
| **FirstRand** | **BNN** | PICP | 0.76 | **0.96** | 0.95 | Calibrated nominal coverage |
| **FirstRand** | **Standard LSTM** | MAE | 769.98 | **82.70** | 89.14 | Outperforms paper baseline |
| **FirstRand** | **ARIMA** | MAE | 1,585.71 | **81.65** | 198.43 | 1-step rolling walk-forward |
| **Discovery** | **GPR** | MAE | 904.21 | **166.42** | 115.15 | -82% error reduction |
| **Discovery** | **GPR** | RMSE | 1,519.99 | **236.29** | 226.19 | Closely matches paper RMSE (226 vs 236) |
| **Discovery** | **GPR** | PICP | 0.62 | **0.98** | 0.92 | Robust interval coverage |
| **Discovery** | **Bayesian LSTM** | MAE | 1,090.08 | **168.99** | 337.39 | -84% error reduction |
| **Discovery** | **Bayesian LSTM** | RMSE | 1,631.11 | **239.15** | 452.78 | -85% error reduction |
| **Discovery** | **Bayesian LSTM** | PICP | 0.72 | **0.97** | 0.95 | Calibrated nominal coverage |
| **Discovery** | **BNN** | MAE | 956.78 | **170.20** | 144.56 | -82% error reduction |
| **Discovery** | **BNN** | RMSE | 1,069.61 | **239.99** | 201.50 | -77% error reduction |
| **Discovery** | **BNN** | PICP | 0.44 | **0.97** | 0.94 | Calibrated nominal coverage |
| **Discovery** | **Standard LSTM** | MAE | 726.91 | **170.02** | 189.73 | Outperforms paper baseline |
| **Discovery** | **ARIMA** | MAE | 3,000.48 | **166.75** | 412.56 | 1-step rolling walk-forward |

---

## 🛠️ Hyperparameters & Methodological Deviations

All changes and tuning configurations are documented in detail below and in [`MODEL_REPRODUCTION_MASTER_DOCUMENT.txt`](file:///c:/Users/rishi/OneDrive/Desktop/pr%20final/MODEL_REPRODUCTION_MASTER_DOCUMENT.txt):

1. **Target Formulation & Causal Price Reconstruction (Deviation #1)**:
   - *Issue in Baseline*: The test period (2021–2025) experiences price levels far above the training domain (FirstRand train max 5,767 vs test max 8,679; Discovery train max 15,450 vs test max 21,178). Regressing raw normalized price levels leads to extrapolation breakdown.
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
pr final/
│
├── data_processing.py                  # Data loading, Wilder RSI, technical indicators, Z-score scaling, windowing
├── models.py                           # Architectures: BNN (BBB), Bayesian LSTM (MC Dropout), GPR, Standard LSTM, ARIMA
├── metrics.py                          # Point metrics (MAE, RMSE, MSE) and uncertainty metrics (PICP, MPIW)
├── plots.py                            # Matplotlib visualizer generating Figures 1–10
├── train_and_evaluate.py               # End-to-end 10-seed training and evaluation pipeline
├── generate_paper_results.py           # Dynamically generates Tables 2–12, Figures 1–10, and comparison reports
├── verify_reproduction.py              # Automated test suite auditing integrity and recomputing metrics
│
├── generate_20_slides_presentation.py  # Presentation generator producing exact 20-slide research defense
├── Bayesian_Stock_Prediction_20_Slides.pptx # High-resolution 20-slide widescreen research defense presentation
├── build_full_notebook.py              # Script to build and execute the Jupyter notebook from scratch
├── Bayesian_Stock_Prediction_Notebook.ipynb # Fully executed Jupyter notebook with math, code, and outputs
│
├── MODEL_REPRODUCTION_MASTER_DOCUMENT.txt # Comprehensive technical manual, equations, and defense documentation
├── MODEL_VS_PAPER_COMPARISON.md        # Three-way comparison analysis: Baseline vs Tuned vs Paper Target
├── paper_settings.md                   # Reference specifications: stated vs unstated paper settings
├── paper_reported_results.csv          # Published benchmark figures from the original research paper
├── algorithms-19-00391.pdf             # Original research article (Nelufhangani & Maposa, 2026)
│
├── FirstRand.csv                       # Historical daily market data for FirstRand Limited (FSR.JO, 2005–2025)
├── Discovery.csv                       # Historical daily market data for Discovery Limited (DSY.JO, 2005–2025)
│
└── outputs/                            # Genuine generated figures, tables, .npz predictions, and markdown reports
    ├── Figure_1_FirstRand_Stock_Closing_Prices.png
    ├── Figure_2_Discovery_Stock_Closing_Prices.png
    ├── Figure_3_FirstRand_Correlation_Matrix.png
    ├── Figure_4_Discovery_Correlation_Matrix.png
    ├── Figure_5_FirstRand_GPR_Forecast.png
    ├── Figure_6_Discovery_GPR_Forecast.png
    ├── Figure_7_FirstRand_Bayesian_LSTM_Forecast.png
    ├── Figure_8_Discovery_Bayesian_LSTM_Forecast.png
    ├── Figure_9_FirstRand_BNN_Forecast.png
    ├── Figure_10_Discovery_BNN_Forecast.png
    ├── Figure_Trained_All_Models_FirstRand.png
    ├── Figure_Trained_All_Models_Discovery.png
    ├── Table_2_FirstRand_Descriptive_Statistics.csv
    ├── Table_3_Discovery_Descriptive_Statistics.csv
    ├── Table_4_GPR_Performance_Metrics.csv
    ├── Table_5_GPR_Actual_vs_Predicted.csv
    ├── Table_6_Bayesian_LSTM_Performance_Metrics.csv
    ├── Table_7_Bayesian_LSTM_Actual_vs_Predicted.csv
    ├── Table_8_BNN_Performance_Metrics.csv
    ├── Table_9_BNN_Actual_vs_Predicted.csv
    ├── Table_10_Comparative_Performance_Bayesian_Models.csv
    ├── Table_11_Uncertainty_Quantification_Metrics.csv
    ├── Table_12_Comparison_Non_Bayesian_Baselines.csv
    ├── Trained_Metrics_FirstRand.csv
    ├── Trained_Metrics_Discovery.csv
    ├── Trained_Predictions_FirstRand.csv
    ├── Trained_Predictions_Discovery.csv
    ├── FirstRand_predictions.npz
    ├── Discovery_predictions.npz
    ├── model_evaluation_metrics.json
    ├── baseline_vs_tuned_vs_paper_comparison.csv
    ├── reproduction_vs_paper_comparison.csv
    ├── reproduction_summary.md
    └── tuning_log.csv
```

---

## 🔒 Verification & Reproducibility Guarantee

1. **Zero Hardcoded Metrics**: All tables, CSV outputs, and markdown summaries are dynamically computed from raw prediction arrays.
2. **Zero Injected Synthetic Noise**: No `np.random.normal` or artificial smoothing is applied to fake prediction intervals.
3. **Strict Validation Discipline**: Hyperparameter tuning was conducted exclusively on the validation partition (2017–2021); the test partition (2021–2025) was evaluated strictly on final calibrated models.
