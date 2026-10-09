"""
Model Implementations Module
Replicating model architectures from:
'Bayesian Deep Learning and Probabilistic Forecasting of Stock Prices'
(Algorithms 2026, 19, 391)

Includes:
1. Gaussian Process Regression (GPR) with composite RBF + WhiteKernel
2. Bayesian Neural Network (BNN) with Bayes by Backprop (ELBO, Gaussian prior)
3. Bayesian LSTM with Monte Carlo Dropout (p=0.2)
4. Standard LSTM (Deterministic baseline)
5. ARIMA Baseline
"""

from typing import cast
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# ==============================================================================
# 1. Bayesian Neural Network (BNN) with Bayes by Backprop (BBB)
# ==============================================================================

class VariationalLinear(nn.Module):
    """
    Bayesian Linear Layer with Gaussian prior N(0, 1) and variational posterior
    q(w) = N(mu, sigma^2), where sigma = softplus(rho).
    Implements Bayes by Backprop (Blundell et al., 2015).
    """
    def __init__(self, in_features: int, out_features: int, prior_std: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std
        
        # Variational parameters for weights: mu and rho
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features).uniform_(-0.05, 0.05))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features).uniform_(-4.0, -3.0))
        
        # Variational parameters for biases: mu and rho
        self.bias_mu = nn.Parameter(torch.empty(out_features).uniform_(-0.05, 0.05))
        self.bias_rho = nn.Parameter(torch.empty(out_features).uniform_(-4.0, -3.0))

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        if sample:
            weight_sigma = F.softplus(self.weight_rho)
            bias_sigma = F.softplus(self.bias_rho)
            w_eps = torch.randn_like(self.weight_mu)
            b_eps = torch.randn_like(self.bias_mu)
            w = self.weight_mu + weight_sigma * w_eps
            b = self.bias_mu + bias_sigma * b_eps
        else:
            w = self.weight_mu
            b = self.bias_mu
        return F.linear(x, w, b)

    def kl_loss(self) -> torch.Tensor:
        """
        Analytical KL divergence KL(q(w) || p(w)) for Gaussian distributions:
        p(w) = N(0, sigma_p^2), q(w) = N(mu, sigma_q^2).
        """
        w_sig = F.softplus(self.weight_rho)
        b_sig = F.softplus(self.bias_rho)
        sig_p = self.prior_std
        
        kl_w = torch.log(sig_p / (w_sig + 1e-8)) + (w_sig**2 + self.weight_mu**2) / (2.0 * sig_p**2) - 0.5
        kl_b = torch.log(sig_p / (b_sig + 1e-8)) + (b_sig**2 + self.bias_mu**2) / (2.0 * sig_p**2) - 0.5
        return kl_w.sum() + kl_b.sum()

class BayesianNeuralNetwork(nn.Module):
    """
    BNN Architecture (Section 2.3):
    - Input: 660 features (60 steps flattened * 11 features)
    - Hidden Layer 1: 64 neurons, ReLU
    - Dropout: rate = 0.1
    - Hidden Layer 2: 32 neurons, ReLU
    - Dropout: rate = 0.1
    - Output Layer: 1 neuron (linear)
    - Prior: Gaussian N(0, 1) over all weights and biases
    """
    def __init__(self, in_features: int = 660, dropout_rate: float = 0.1, prior_std: float = 1.0):
        super().__init__()
        self.fc1 = VariationalLinear(in_features, 64, prior_std=prior_std)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc2 = VariationalLinear(64, 32, prior_std=prior_std)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.out = VariationalLinear(32, 1, prior_std=prior_std)

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        h1 = F.relu(self.fc1(x, sample=sample))
        h1 = self.dropout1(h1)
        h2 = F.relu(self.fc2(h1, sample=sample))
        h2 = self.dropout2(h2)
        return self.out(h2, sample=sample)

    def total_kl(self) -> torch.Tensor:
        return self.fc1.kl_loss() + self.fc2.kl_loss() + self.out.kl_loss()

    def predict_monte_carlo(self, x: np.ndarray, s_samples: int = 100, device: str = "cpu"):
        """
        Draws S Monte Carlo samples from the approximate posterior to estimate
        predictive mean and standard deviation (Equation 7).
        """
        self.eval()
        x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
        preds = []
        with torch.no_grad():
            for _ in range(s_samples):
                y_s = self.forward(x_tensor, sample=True).squeeze(-1)
                preds.append(y_s.cpu().numpy())
        
        preds = np.stack(preds, axis=0) # Shape: (S, N)
        mean = np.mean(preds, axis=0)
        variance = np.var(preds, axis=0, ddof=1)
        std = np.sqrt(variance)
        return mean, std

# ==============================================================================
# 2. Bayesian LSTM with Monte Carlo Dropout
# ==============================================================================

class BayesianLSTM(nn.Module):
    """
    Bayesian LSTM Architecture (Section 2.4):
    - Input: Sequence window of 60 time steps, 11 features
    - Stacked LSTM 1: 50 units
    - Dropout: rate = 0.2 (applied at both train and test time)
    - Stacked LSTM 2: 50 units
    - Dropout: rate = 0.2 (applied at both train and test time)
    - Dense output layer: 1 unit
    """
    def __init__(self, input_size: int = 11, hidden_units: int = 50, dropout_rate: float = 0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_units, batch_first=True)
        self.dropout_rate = dropout_rate
        self.lstm2 = nn.LSTM(hidden_units, hidden_units, batch_first=True)
        self.fc = nn.Linear(hidden_units, 1)

    def forward(self, x: torch.Tensor, force_dropout: bool = True) -> torch.Tensor:
        out1, _ = self.lstm1(x)
        if force_dropout or self.training:
            out1 = F.dropout(out1, p=self.dropout_rate, training=True)
            
        out2, _ = self.lstm2(out1)
        last_step = out2[:, -1, :] # Last time step output
        
        if force_dropout or self.training:
            last_step = F.dropout(last_step, p=self.dropout_rate, training=True)
            
        return self.fc(last_step)

    def predict_monte_carlo(self, x: np.ndarray, s_samples: int = 100, device: str = "cpu"):
        """
        S = 100 Monte Carlo forward passes with dropout active at test time (Section 2.4).
        """
        self.eval()
        x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
        preds = []
        with torch.no_grad():
            for _ in range(s_samples):
                y_s = self.forward(x_tensor, force_dropout=True).squeeze(-1)
                preds.append(y_s.cpu().numpy())
                
        preds = np.stack(preds, axis=0) # Shape: (S, N)
        mean = np.mean(preds, axis=0)
        variance = np.var(preds, axis=0, ddof=1)
        std = np.sqrt(variance)
        return mean, std

# ==============================================================================
# 3. Gaussian Process Regression (GPR)
# ==============================================================================

class GPRModel:
    """
    Gaussian Process Regression (Section 2.5):
    - Composite Kernel: RBF + WhiteKernel (residual noise)
    - Hyperparameters optimized by maximizing log marginal likelihood using L-BFGS-B
      with five random restarts.
    - Output: predictive mean and 95% predictive confidence intervals.
    """
    def __init__(self, length_scale: float = 1.0, noise_level: float = 1e-3, n_restarts: int = 5):
        kernel = RBF(length_scale=length_scale, length_scale_bounds=(1e-2, 1e3)) + \
                 WhiteKernel(noise_level=noise_level, noise_level_bounds=(1e-5, 1e1))
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=n_restarts,
            random_state=42,
            normalize_y=False
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Flattened inputs (N, 660)
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean, std = cast(tuple[np.ndarray, np.ndarray], self.model.predict(X, return_std=True))
        return mean, std

# ==============================================================================
# 4. Deterministic Baselines: Standard LSTM and ARIMA
# ==============================================================================

class StandardLSTM(nn.Module):
    """
    Standard Deterministic LSTM baseline (Table 12):
    - Two stacked LSTM layers with 50 units each
    - Dense output layer
    - Standard dropout at train time only, deterministic inference.
    """
    def __init__(self, input_size: int = 11, hidden_units: int = 50, dropout_rate: float = 0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_units, batch_first=True)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.lstm2 = nn.LSTM(hidden_units, hidden_units, batch_first=True)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(hidden_units, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1, _ = self.lstm1(x)
        out1 = self.dropout1(out1)
        out2, _ = self.lstm2(out1)
        last_step = self.dropout2(out2[:, -1, :])
        return self.fc(last_step)

    def predict(self, x: np.ndarray, device: str = "cpu") -> np.ndarray:
        self.eval()
        x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
        with torch.no_grad():
            preds = self.forward(x_tensor).squeeze(-1)
        return preds.cpu().numpy()

class ARIMABaseline:
    """
    ARIMA baseline (Section 1.2, Table 12):
    Uses statsmodels ARIMA model fitted on training price levels with 1-step walk-forward rolling inference.
    """
    def __init__(self, order=(1, 1, 1)):
        self.order = order
        self.fitted_model = None

    def fit_and_predict(self, train_series: np.ndarray, steps: int):
        from statsmodels.tsa.arima.model import ARIMA
        model = ARIMA(train_series, order=self.order)
        self.fitted_model = model.fit()
        forecast = self.fitted_model.forecast(steps=steps)
        return forecast

    def fit_and_predict_rolling(self, full_series: np.ndarray, train_len: int, eval_len: int) -> np.ndarray:
        from statsmodels.tsa.arima.model import ARIMA
        train_series = full_series[:train_len]
        model = ARIMA(train_series, order=self.order)
        self.fitted_model = model.fit()
        eval_series = full_series[:train_len + eval_len]
        res_full = ARIMA(eval_series, order=self.order).filter(self.fitted_model.params)
        forecasts = res_full.predict(start=train_len, end=train_len + eval_len - 1)
        return np.array(forecasts, dtype=np.float64)
