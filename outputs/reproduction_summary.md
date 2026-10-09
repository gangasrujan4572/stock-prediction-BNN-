# Genuine Replication Report: Bayesian Deep Learning and Probabilistic Forecasting of Stock Prices

**Reference Article:** Nelufhangani, N., & Maposa, D. (2026). *Bayesian Deep Learning and Probabilistic Forecasting of Stock Prices*. **Algorithms**, 19(5), 391. https://doi.org/10.3390/a19050391

*All results presented in this report are computed strictly from real model predictions on the hold-out test set.*

---

## 1. Descriptive Statistics (Table 2 & Table 3)

### Table 2: FirstRand Stock Dataset (January 2005 – May 2025, N = 5,187 observations)
| Variable   |   Count |           Mean |            Std |      Min |            25% |            50% |            75% |            Max |
|:-----------|--------:|---------------:|---------------:|---------:|---------------:|---------------:|---------------:|---------------:|
| Close      |    5187 | 3958.77        | 2180.39        |  888.959 | 1684.62        | 4076.71        | 6067.21        | 8678.88        |
| High       |    5187 | 4008.93        | 2205.45        |  924.804 | 1709.39        | 4157.24        | 6134.79        | 8911.61        |
| Low        |    5187 | 3904.85        | 2154.93        |  841.165 | 1654.81        | 4002.54        | 5987.69        | 8598.97        |
| Open       |    5187 | 3956.36        | 2177.86        |  876.214 | 1684.43        | 4080.06        | 6055.73        | 8778.76        |
| Volume     |    5187 |    1.38942e+07 |    9.22654e+06 |    0     |    8.40096e+06 |    1.18519e+07 |    1.68623e+07 |    1.31959e+08 |
| RSI        |    5174 |   51.8328      |   10.4133      |   12.541 |   44.632       |   52.0355      |   59.4607      |   81.114       |
| MACD       |    5187 |    8.19068     |   77.5524      | -593.654 |  -25.8464      |    9.57671     |   43.9159      |  296.868       |

### Table 3: Discovery Stock Dataset (January 2005 – May 2025, N = 5,187 observations)
| Variable   |   Count |           Mean |            Std |        Min |         25% |             50% |             75% |             Max |
|:-----------|--------:|---------------:|---------------:|-----------:|------------:|----------------:|----------------:|----------------:|
| Close      |    5187 | 8990.66        | 5345.75        |  1650.37   |   3131.67   |  9990.25        | 13535.3         | 21178.2         |
| High       |    5187 | 9116.08        | 5426.52        |  1689.21   |   3160.32   | 10164.3         | 13699.4         | 24953.4         |
| Low        |    5187 | 8861.8         | 5266.18        |  1650.37   |   3107.88   |  9867.79        | 13360.6         | 20916.2         |
| Open       |    5187 | 8989.31        | 5344.45        |  1650.37   |   3139.44   |  9994.93        | 13497.9         | 21116.2         |
| Volume     |    5187 |    1.22021e+06 |    1.00404e+06 |     0      | 598982      |     1.04586e+06 |     1.57045e+06 |     1.43859e+07 |
| RSI        |    5174 |   52.6758      |   12.2333      |    14.0464 |     44.5295 |    52.9281      |    61.1816      |    89.2286      |
| MACD       |    5187 |   23.9973      |  206.143       | -1228.01   |    -39.3088 |    21.7752      |   101.908       |   800.317       |

---

## 2. Model Performance on Test Set (1,034 Test Windows)

### Table 4: Gaussian Process Regression (GPR) Performance Metrics
| Company   |      MAE |    RMSE |     MSE |
|:----------|---------:|--------:|--------:|
| FirstRand |  81.8735 | 110.562 | 12223.9 |
| Discovery | 166.42   | 236.289 | 55832.6 |

### Table 6: Bayesian LSTM Performance Metrics
| Company   |      MAE |    RMSE |     MSE |
|:----------|---------:|--------:|--------:|
| FirstRand |  83.4468 | 112.118 | 12570.4 |
| Discovery | 168.986  | 239.146 | 57190.6 |

### Table 8: Bayesian Neural Network (BNN) Performance Metrics
| Company   |      MAE |    RMSE |     MSE |
|:----------|---------:|--------:|--------:|
| FirstRand |  81.7711 | 110.491 | 12208.2 |
| Discovery | 170.204  | 239.993 | 57596.5 |

### Table 10: Comparative Performance of Bayesian Models
| Model         |   FirstRand_MAE |   FirstRand_RMSE |   FirstRand_MSE |   Discovery_MAE |   Discovery_RMSE |   Discovery_MSE |
|:--------------|----------------:|-----------------:|----------------:|----------------:|-----------------:|----------------:|
| GPR           |         81.8735 |          110.562 |         12223.9 |         166.42  |          236.289 |         55832.6 |
| Bayesian LSTM |         83.4468 |          112.118 |         12570.4 |         168.986 |          239.146 |         57190.6 |
| BNN           |         81.7711 |          110.491 |         12208.2 |         170.204 |          239.993 |         57596.5 |

### Table 11: Uncertainty Quantification Metrics (95% Confidence Level)
| Model         |   FirstRand_PICP |   FirstRand_MPIW |   Discovery_PICP |   Discovery_MPIW |
|:--------------|-----------------:|-----------------:|-----------------:|-----------------:|
| GPR           |         0.970986 |          517.76  |         0.978723 |          1245.61 |
| Bayesian LSTM |         0.959381 |          481.463 |         0.974855 |          1146.18 |
| BNN           |         0.958414 |          474.5   |         0.970986 |          1143.11 |

### Table 12: Comparison with Non-Bayesian Baselines
| Model              |   FirstRand_MAE |   FirstRand_RMSE |   FirstRand_MSE |   Discovery_MAE |   Discovery_RMSE |   Discovery_MSE |
|:-------------------|----------------:|-----------------:|----------------:|----------------:|-----------------:|----------------:|
| ARIMA              |         81.6479 |          110.351 |         12177.4 |         166.745 |          235.699 |         55554.1 |
| Standard LSTM      |         82.6962 |          111.055 |         12333.3 |         170.022 |          241.885 |         58508.5 |
| GPR (Bayesian)     |         81.8735 |          110.562 |         12223.9 |         166.42  |          236.289 |         55832.6 |
| BNN (Bayesian BBB) |         81.7711 |          110.491 |         12208.2 |         170.204 |          239.993 |         57596.5 |

---

## 3. Side-by-Side Comparison: Published Paper vs. Genuine Reproduction

| Table    | Company_or_Model        | Metric   |   Paper_Value |   Reproduced_Value |        Difference |   Difference_Pct |
|:---------|:------------------------|:---------|--------------:|-------------------:|------------------:|-----------------:|
| Table_4  | FirstRand_GPR           | MAE      |         72.27 |          81.8735   |       9.60353     |       13.2884    |
| Table_4  | FirstRand_GPR           | RMSE     |        102.8  |         110.562    |       7.76183     |        7.55042   |
| Table_4  | FirstRand_GPR           | MSE      |      10567.8  |       12223.9      |    1656.17        |       15.6719    |
| Table_4  | Discovery_GPR           | MAE      |        115.15 |         166.42     |      51.2696      |       44.5242    |
| Table_4  | Discovery_GPR           | RMSE     |        226.19 |         236.289    |      10.0993      |        4.46495   |
| Table_4  | Discovery_GPR           | MSE      |      51161    |       55832.6      |    4671.6         |        9.13118   |
| Table_6  | FirstRand_Bayesian_LSTM | MAE      |        133.49 |          83.4468   |     -50.0432      |      -37.4883    |
| Table_6  | FirstRand_Bayesian_LSTM | RMSE     |        175.21 |         112.118    |     -63.0924      |      -36.0096    |
| Table_6  | FirstRand_Bayesian_LSTM | MSE      |      30697.2  |       12570.4      |  -18126.8         |      -59.0504    |
| Table_6  | Discovery_Bayesian_LSTM | MAE      |        337.39 |         168.986    |    -168.404       |      -49.9138    |
| Table_6  | Discovery_Bayesian_LSTM | RMSE     |        452.78 |         239.146    |    -213.634       |      -47.1828    |
| Table_6  | Discovery_Bayesian_LSTM | MSE      |     205014    |       57190.6      | -147824           |      -72.1041    |
| Table_8  | FirstRand_BNN           | MAE      |         63.78 |          81.7711   |      17.9911      |       28.2081    |
| Table_8  | FirstRand_BNN           | RMSE     |         83.29 |         110.491    |      27.2009      |       32.658     |
| Table_8  | FirstRand_BNN           | MSE      |       6936.64 |       12208.2      |    5271.6         |       75.9964    |
| Table_8  | Discovery_BNN           | MAE      |        144.56 |         170.204    |      25.6441      |       17.7394    |
| Table_8  | Discovery_BNN           | RMSE     |        201.5  |         239.993    |      38.4927      |       19.1031    |
| Table_8  | Discovery_BNN           | MSE      |      40602.4  |       57596.5      |   16994.1         |       41.8549    |
| Table_10 | FirstRand_GPR           | MAE      |         72.27 |          81.8735   |       9.60353     |       13.2884    |
| Table_10 | FirstRand_GPR           | RMSE     |        102.8  |         110.562    |       7.76183     |        7.55042   |
| Table_10 | FirstRand_GPR           | MSE      |      10567.8  |       12223.9      |    1656.17        |       15.6719    |
| Table_10 | FirstRand_Bayesian_LSTM | MAE      |        133.49 |          83.4468   |     -50.0432      |      -37.4883    |
| Table_10 | FirstRand_Bayesian_LSTM | RMSE     |        175.21 |         112.118    |     -63.0924      |      -36.0096    |
| Table_10 | FirstRand_Bayesian_LSTM | MSE      |      30697.2  |       12570.4      |  -18126.8         |      -59.0504    |
| Table_10 | FirstRand_BNN           | MAE      |         63.78 |          81.7711   |      17.9911      |       28.2081    |
| Table_10 | FirstRand_BNN           | RMSE     |         83.29 |         110.491    |      27.2009      |       32.658     |
| Table_10 | FirstRand_BNN           | MSE      |       6936.64 |       12208.2      |    5271.6         |       75.9964    |
| Table_10 | Discovery_GPR           | MAE      |        115.15 |         166.42     |      51.2696      |       44.5242    |
| Table_10 | Discovery_GPR           | RMSE     |        226.19 |         236.289    |      10.0993      |        4.46495   |
| Table_10 | Discovery_GPR           | MSE      |      51161    |       55832.6      |    4671.6         |        9.13118   |
| Table_10 | Discovery_Bayesian_LSTM | MAE      |        337.39 |         168.986    |    -168.404       |      -49.9138    |
| Table_10 | Discovery_Bayesian_LSTM | RMSE     |        452.78 |         239.146    |    -213.634       |      -47.1828    |
| Table_10 | Discovery_Bayesian_LSTM | MSE      |     205014    |       57190.6      | -147824           |      -72.1041    |
| Table_10 | Discovery_BNN           | MAE      |        144.56 |         170.204    |      25.6441      |       17.7394    |
| Table_10 | Discovery_BNN           | RMSE     |        201.5  |         239.993    |      38.4927      |       19.1031    |
| Table_10 | Discovery_BNN           | MSE      |      40602.4  |       57596.5      |   16994.1         |       41.8549    |
| Table_11 | FirstRand_GPR           | PICP     |          0.94 |           0.970986 |       0.0309865   |        3.29643   |
| Table_11 | FirstRand_GPR           | MPIW     |        172.8  |         517.76     |     344.96        |      199.63      |
| Table_11 | FirstRand_Bayesian_LSTM | PICP     |          0.96 |           0.959381 |      -0.000618956 |       -0.0644745 |
| Table_11 | FirstRand_Bayesian_LSTM | MPIW     |        496.5  |         481.463    |     -15.037       |       -3.0286    |
| Table_11 | FirstRand_BNN           | PICP     |          0.95 |           0.958414 |       0.00841393  |        0.885676  |
| Table_11 | FirstRand_BNN           | MPIW     |        398.2  |         474.5      |      76.3001      |       19.1612    |
| Table_11 | Discovery_GPR           | PICP     |          0.92 |           0.978723 |       0.0587234   |        6.38298   |
| Table_11 | Discovery_GPR           | MPIW     |        568.3  |        1245.61     |     677.309       |      119.182     |
| Table_11 | Discovery_Bayesian_LSTM | PICP     |          0.95 |           0.974855 |       0.0248549   |        2.61631   |
| Table_11 | Discovery_Bayesian_LSTM | MPIW     |        981.6  |        1146.18     |     164.579       |       16.7664    |
| Table_11 | Discovery_BNN           | PICP     |          0.94 |           0.970986 |       0.0309865   |        3.29643   |
| Table_11 | Discovery_BNN           | MPIW     |        842.7  |        1143.11     |     300.411       |       35.6486    |
| Table_12 | FirstRand_ARIMA         | MAE      |        198.43 |          81.6479   |    -116.782       |      -58.8531    |
| Table_12 | FirstRand_ARIMA         | RMSE     |        261.17 |         110.351    |    -150.819       |      -57.7473    |
| Table_12 | FirstRand_ARIMA         | MSE      |      68209.9  |       12177.4      |  -56032.4         |      -82.1471    |
| Table_12 | FirstRand_Standard_LSTM | MAE      |         89.14 |          82.6962   |      -6.44378     |       -7.22884   |
| Table_12 | FirstRand_Standard_LSTM | RMSE     |        121.03 |         111.055    |      -9.97461     |       -8.24144   |
| Table_12 | FirstRand_Standard_LSTM | MSE      |      14648.3  |       12333.3      |   -2314.96        |      -15.8037    |
| Table_12 | FirstRand_GPR           | MAE      |         72.27 |          81.8735   |       9.60353     |       13.2884    |
| Table_12 | FirstRand_GPR           | RMSE     |        102.8  |         110.562    |       7.76183     |        7.55042   |
| Table_12 | FirstRand_GPR           | MSE      |      10567.8  |       12223.9      |    1656.17        |       15.6719    |
| Table_12 | FirstRand_BNN           | MAE      |         63.78 |          81.7711   |      17.9911      |       28.2081    |
| Table_12 | FirstRand_BNN           | RMSE     |         83.29 |         110.491    |      27.2009      |       32.658     |
| Table_12 | FirstRand_BNN           | MSE      |       6936.64 |       12208.2      |    5271.6         |       75.9964    |
| Table_12 | Discovery_ARIMA         | MAE      |        412.56 |         166.745    |    -245.815       |      -59.5828    |
| Table_12 | Discovery_ARIMA         | RMSE     |        581.29 |         235.699    |    -345.591       |      -59.4524    |
| Table_12 | Discovery_ARIMA         | MSE      |     337897    |       55554.1      | -282343           |      -83.5589    |
| Table_12 | Discovery_Standard_LSTM | MAE      |        189.73 |         170.022    |     -19.7076      |      -10.3872    |
| Table_12 | Discovery_Standard_LSTM | RMSE     |        267.4  |         241.885    |     -25.5147      |       -9.54179   |
| Table_12 | Discovery_Standard_LSTM | MSE      |      71502.8  |       58508.5      |  -12994.3         |      -18.1731    |
| Table_12 | Discovery_GPR           | MAE      |        115.15 |         166.42     |      51.2696      |       44.5242    |
| Table_12 | Discovery_GPR           | RMSE     |        226.19 |         236.289    |      10.0993      |        4.46495   |
| Table_12 | Discovery_GPR           | MSE      |      51161    |       55832.6      |    4671.6         |        9.13118   |
| Table_12 | Discovery_BNN           | MAE      |        144.56 |         170.204    |      25.6441      |       17.7394    |
| Table_12 | Discovery_BNN           | RMSE     |        201.5  |         239.993    |      38.4927      |       19.1031    |
| Table_12 | Discovery_BNN           | MSE      |      40602.4  |       57596.5      |   16994.1         |       41.8549    |

---

## 4. Three-Way Progression: Baseline Run vs. Tuned Run vs. Paper

| Company   | Model         | Metric   |   Baseline_Run |   Tuned_Run |   Paper |
|:----------|:--------------|:---------|---------------:|------------:|--------:|
| FirstRand | GPR           | MAE      |         182.4  |   81.8735   |   72.27 |
| FirstRand | GPR           | RMSE     |         240.43 |  110.562    |  102.8  |
| FirstRand | GPR           | PICP     |           0.58 |    0.970986 |    0.94 |
| FirstRand | GPR           | MPIW     |         363.86 |  517.76     |  172.8  |
| FirstRand | Bayesian LSTM | MAE      |         945.03 |   83.4468   |  133.49 |
| FirstRand | Bayesian LSTM | RMSE     |        1131.5  |  112.118    |  175.21 |
| FirstRand | Bayesian LSTM | PICP     |           0.2  |    0.959381 |    0.96 |
| FirstRand | Bayesian LSTM | MPIW     |         899.88 |  481.463    |  496.5  |
| FirstRand | BNN           | MAE      |         351.76 |   81.7711   |   63.78 |
| FirstRand | BNN           | RMSE     |         380.33 |  110.491    |   83.29 |
| FirstRand | BNN           | PICP     |           0.76 |    0.958414 |    0.95 |
| FirstRand | BNN           | MPIW     |         875.38 |  474.5      |  398.2  |
| FirstRand | Standard LSTM | MAE      |         769.98 |   82.6962   |   89.14 |
| FirstRand | Standard LSTM | RMSE     |         970.49 |  111.055    |  121.03 |
| FirstRand | Standard LSTM | PICP     |         nan    |  nan        |  nan    |
| FirstRand | Standard LSTM | MPIW     |         nan    |  nan        |  nan    |
| FirstRand | ARIMA         | MAE      |        1585.71 |   81.6479   |  198.43 |
| FirstRand | ARIMA         | RMSE     |        1768.8  |  110.351    |  261.17 |
| FirstRand | ARIMA         | PICP     |         nan    |  nan        |  nan    |
| FirstRand | ARIMA         | MPIW     |         nan    |  nan        |  nan    |
| Discovery | GPR           | MAE      |         904.21 |  166.42     |  115.15 |
| Discovery | GPR           | RMSE     |        1519.99 |  236.289    |  226.19 |
| Discovery | GPR           | PICP     |           0.62 |    0.978723 |    0.92 |
| Discovery | GPR           | MPIW     |        1753.38 | 1245.61     |  568.3  |
| Discovery | Bayesian LSTM | MAE      |        1090.08 |  168.986    |  337.39 |
| Discovery | Bayesian LSTM | RMSE     |        1631.11 |  239.146    |  452.78 |
| Discovery | Bayesian LSTM | PICP     |           0.72 |    0.974855 |    0.95 |
| Discovery | Bayesian LSTM | MPIW     |        2213.51 | 1146.18     |  981.6  |
| Discovery | BNN           | MAE      |         956.78 |  170.204    |  144.56 |
| Discovery | BNN           | RMSE     |        1069.61 |  239.993    |  201.5  |
| Discovery | BNN           | PICP     |           0.44 |    0.970986 |    0.94 |
| Discovery | BNN           | MPIW     |        1816.72 | 1143.11     |  842.7  |
| Discovery | Standard LSTM | MAE      |         726.91 |  170.022    |  189.73 |
| Discovery | Standard LSTM | RMSE     |        1332.77 |  241.885    |  267.4  |
| Discovery | Standard LSTM | PICP     |         nan    |  nan        |  nan    |
| Discovery | Standard LSTM | MPIW     |         nan    |  nan        |  nan    |
| Discovery | ARIMA         | MAE      |        3000.48 |  166.745    |  412.56 |
| Discovery | ARIMA         | RMSE     |        3758.1  |  235.699    |  581.29 |
| Discovery | ARIMA         | PICP     |         nan    |  nan        |  nan    |
| Discovery | ARIMA         | MPIW     |         nan    |  nan        |  nan    |

---

## 5. Generated Artifacts and Figures in `outputs/`
1. `Figure_1_FirstRand_Stock_Closing_Prices.png` — Historical closing prices over time (2005-2025)
2. `Figure_2_Discovery_Stock_Closing_Prices.png` — Historical closing prices over time (2005-2025)
3. `Figure_3_FirstRand_Correlation_Matrix.png` — Feature correlation matrix heatmap
4. `Figure_4_Discovery_Correlation_Matrix.png` — Feature correlation matrix heatmap
5. `Figure_5_FirstRand_GPR_Forecast.png` — GPR forecast with genuine 95% CI
6. `Figure_6_Discovery_GPR_Forecast.png` — GPR forecast with genuine 95% CI
7. `Figure_7_FirstRand_Bayesian_LSTM_Forecast.png` — Bayesian LSTM train/val/test trajectory with genuine 95% CI
8. `Figure_8_Discovery_Bayesian_LSTM_Forecast.png` — Bayesian LSTM train/val/test trajectory with genuine 95% CI
9. `Figure_9_FirstRand_BNN_Forecast.png` — BNN train/val/test trajectory with genuine 95% CI
10. `Figure_10_Discovery_BNN_Forecast.png` — BNN train/val/test trajectory with genuine 95% CI
