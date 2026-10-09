# Research Paper Settings & Specification Reference

**Paper:** *Bayesian Deep Learning and Probabilistic Forecasting of Stock Prices*  
**Journal:** *Algorithms* 2026, 19(5), 391  
**Target Stocks:** FirstRand Limited (FSR) and Discovery Limited (DSY) (JSE equities, Jan 2005 – June 2025)

---

## 1. Explicitly Stated Settings in the Paper

| Category | Component | Stated Value / Configuration | Paper Section / Reference |
| :--- | :--- | :--- | :--- |
| **Data** | Timeframe | January 2005 – June 2025 | Section 2.1 |
| **Data** | Total Observations | 5,187 daily observations (OHLCV) | Section 2.1, Table 2 & 3 |
| **Data** | Cleaned Observations | 5,167 usable observations (first 20 rows dropped for indicator warmup) | Section 2.2 |
| **Data** | Split Ratios | 60% Train (3,100), 20% Validation (1,033), 20% Test (1,034) | Section 2.2 |
| **Features** | 11 Features | Open, High, Low, Close, Volume, RSI(14), MACD(12,26 line), SMA-20, EMA-20, BB-Upper(20,2), BB-Lower(20,2) | Section 2.2 |
| **Scaling** | Standardization | Z-score scaling: $z = (x - \mu)/\sigma$, computed **exclusively** on training set | Section 2.2, Eq. (1)–(2) |
| **Input Shape** | Lookback Window | 60 consecutive daily steps ($W = 60$) | Section 2.2 |
| **Input Shape** | LSTM input | Sequential 3D tensor: $(N, 60, 11)$ | Section 2.2 |
| **Input Shape** | BNN & GPR input | Flattened 2D vector: $(N, 660)$ where $660 = 60 \times 11$ | Section 2.2 |
| **Target** | Target Variable | Next-day Closing price ($y_{t+1}$) | Section 2.2 |
| **BNN** | Architecture | 2 FC hidden layers: FC(660 $\to$ 64, ReLU) $\to$ FC(64 $\to$ 32, ReLU) $\to$ FC(32 $\to$ 1, Linear) | Section 2.3 |
| **BNN** | Inference Method | Bayes by Backprop (BBB) variational inference, ELBO objective | Section 2.3, Eq. (6) |
| **BNN** | Prior Distribution | Standard Gaussian $\mathcal{N}(0, 1)$ over all weights and biases | Section 2.3, Eq. (3) |
| **BNN** | Optimizer & LR | Adam optimizer, learning rate = 0.001 | Section 2.3 |
| **BNN** | Training Budget | 200 epochs, batch size = 32 | Section 2.3 |
| **BNN** | Regularization | Dropout rate = 0.1 | Section 2.3 |
| **BNN** | MC Samples | $S = 100$ Monte Carlo posterior samples at test time | Section 2.3, Eq. (7) |
| **Bayesian LSTM** | Architecture | 2 stacked LSTM layers with 50 hidden units each $\to$ Dense(1) | Section 2.4 |
| **Bayesian LSTM** | Inference Method | Monte Carlo Dropout (dropout rate = 0.2 active during train & test) | Section 2.4 |
| **Bayesian LSTM** | Optimizer & LR | Adam optimizer, learning rate = 0.001, MSE loss | Section 2.4 |
| **Bayesian LSTM** | Training Budget | 100 epochs, batch size = 32 | Section 2.4 |
| **Bayesian LSTM** | MC Samples | $S = 100$ stochastic forward passes | Section 2.4 |
| **GPR** | Kernel | Composite: RBF + WhiteKernel ($k(x, x') = \sigma_f^2 \exp(-\frac{\|x-x'\|^2}{2\ell^2}) + \sigma_n^2 \delta(x,x')$) | Section 2.5 |
| **GPR** | Optimization | L-BFGS-B maximizing log marginal likelihood, 5 random restarts | Section 2.5, Eq. (16) |
| **GPR** | Interval Formula | $\mu^* \pm 1.96 \sqrt{\operatorname{diag}(\Sigma^* + \sigma_\epsilon^2 I)}$ | Section 2.5, Eq. (15) |
| **Metrics** | Point Metrics | MAE, RMSE, MSE | Section 2.6, Eq. (17)–(19) |
| **Metrics** | UQ Metrics | PICP (nominal 95%), MPIW (mean interval width $\hat{u} - \hat{l}$) | Section 2.7, Eq. (20)–(21) |

---

## 2. Unstated / Ambiguous Settings in the Paper

1. **BNN ELBO Weighting / Annealing ($\beta$ KL weight):**
   - The paper states the ELBO formula $\mathcal{L}(\theta) = \mathbb{E}[\log p(D|w)] - \text{KL}(q(w) \| p(w))$ but does not state the minibatch scale factor $\beta$. In stochastic gradient descent, typical scaling is $\beta = 1/N_{\text{train}}$ or $1/N_{\text{batches}}$ or a tuned hyperparameter.
2. **Predictive Uncertainty / Residual Noise for Neural Models:**
   - MC Dropout and standard BBB yield epistemic uncertainty $\sigma_{\text{epistemic}} = \text{std}(\hat{y}^{(s)})$. Without an observational noise term $\sigma_{\text{aleatoric}}$ (or calibrated temperature scaling), raw epistemic intervals capture model parameter variance only, resulting in severe PICP under-coverage on out-of-sample data.
3. **ARIMA Order and Forecasting Mode:**
   - The paper does not specify the $(p, d, q)$ order for FirstRand or Discovery, nor whether forecasts are produced via a single multi-step trajectory or rolling 1-step-ahead walk-forward.
4. **Standard LSTM baseline details:**
   - Baseline LSTM architecture is described contextually without separate optimizer/epoch details (inferred as identical to Bayesian LSTM without test-time MC dropout).
5. **Handling Out-of-Distribution Price Range:**
   - In 2021–2025 (the test split), stock prices climbed substantially beyond the 2005–2017 training range (FirstRand max 5,767 in train vs 8,679 in test; Discovery max 15,450 in train vs 21,178 in test). Standard Z-score scaling on raw prices presents $z$-values up to $+3.5\sigma$ that neural networks and stationary RBF kernels cannot extrapolate without relative residual/change targeting.

---

## 3. Reproduction Policy & Rules

1. **Validation-Only Tuning:** All parameter choices, early stopping criteria, KL weights, and residual calibration factors must be selected strictly on the validation set (indices 3,100 to 4,133). The test set (indices 4,134 to 5,167) is evaluated only once for final reporting.
2. **Honesty & Integrity:** Zero synthetic noise, zero hardcoded metric values, zero test leakage.
3. **Deviations:** Any methodological adaptation required to handle out-of-distribution price levels or uncertainty calibration is explicitly listed as a deviation.
