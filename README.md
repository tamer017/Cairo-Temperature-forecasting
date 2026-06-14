# Cairo Temperature Forecasting

> **Production-quality XGBoost forecasting pipeline for Cairo daily maximum temperature achieving RMSE of 2.3°C using lag features, Fourier seasonality terms, and rolling statistics.**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-orange.svg)](https://xgboost.readthedocs.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-yellow.svg)](https://jupyter.org/)

---

## Overview

This project builds a rigorous time-series forecasting pipeline for **Cairo daily maximum temperature** using gradient boosting. The pipeline prioritizes methodological correctness: a chronological train/test split avoids any lookahead bias, and all features are derived from observable historical data only.

---

## Feature Engineering

| Feature Group | Features | Purpose |
|---|---|---|
| **Lag features** | T-1, T-2, T-3, T-7, T-14, T-30 | Recent temperature memory |
| **Rolling statistics** | 7-day mean/std, 30-day mean/std | Smoothed trend estimation |
| **Fourier terms** | sin/cos (period=365.25 days) | Annual seasonality decomposition |
| **Calendar features** | Day of year, month, week, weekday | Explicit seasonality encoding |
| **Outlier removal** | IQR-based ±3σ filter | Robustness to sensor anomalies |

### Fourier Seasonality Encoding
```python
# Capture annual temperature cycle without assuming linear trends
df['sin_annual'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
df['cos_annual'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
# Add harmonics for finer seasonal patterns
df['sin_semi'] = np.sin(4 * np.pi * df['day_of_year'] / 365.25)
df['cos_semi'] = np.cos(4 * np.pi * df['day_of_year'] / 365.25)
```

---

## Results

| Metric | Value |
|---|---|
| **Test RMSE** | **2.3°C** |
| **MAE** | 1.7°C |
| Split method | Chronological (no shuffle) |
| Training period | 1990–2019 |
| Test period | 2020–2022 |

**Note:** Chronological splitting is critical for time-series — random splitting would artificially inflate performance by leaking future information.

---

## Model

```python
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=50
)
```

---

## Installation

```bash
git clone https://github.com/tamer017/Cairo-Temperature-forecasting.git
cd Cairo-Temperature-forecasting
pip install xgboost pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook
```

---

## Skills & Concepts

`Time-Series Forecasting` `XGBoost` `Fourier Feature Engineering` `Chronological Cross-Validation` `Seasonality Modeling` `Lag Features` `Rolling Statistics` `Climate Analytics` `Gradient Boosting`

---

## Author

**Ahmed Tamer Assy** — [GitHub](https://github.com/tamer017) | Machine Learning Researcher @ Volkswagen AG
