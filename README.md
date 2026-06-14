# Cairo Temperature Forecasting — XGBoost Time-Series Pipeline

> A production-quality time-series forecasting pipeline for **Cairo daily maximum temperature** prediction using gradient boosting with extensive feature engineering.

[![Model](https://img.shields.io/badge/Model-XGBoost-blue?style=flat-square)](https://xgboost.readthedocs.io/)
[![RMSE](https://img.shields.io/badge/Best%20RMSE-2.3%C2%B0C-brightgreen?style=flat-square)]()
[![Language](https://img.shields.io/badge/Language-Python%203.x-green?style=flat-square)](https://www.python.org/)

---

## Overview

This project builds a **time-series temperature forecasting pipeline** for predicting Cairo’s daily maximum temperature. It combines robust data preprocessing, advanced feature engineering (lag features, rolling statistics, Fourier terms), and an **XGBoost gradient boosting model** to achieve a best **RMSE of 2.3°C** on the hold-out test set.

The pipeline is designed as a reproducible, modular system with clean separation between data ingestion, feature engineering, model training, and evaluation — following production ML engineering best practices.

---

## Dataset

| Property | Detail |
|---|---|
| Source | Cairo climate records |
| Target variable | Daily maximum temperature (°C) |
| Date range | Multi-year historical records |
| Preprocessing | IQR-based outlier removal, forward-fill for missing values |

---

## Feature Engineering

The model relies entirely on **lag and derived features** (no raw future data leakage):

| Feature Type | Description |
|---|---|
| **Lag features** | T-1, T-2, T-3, T-7, T-14, T-30 day temperature lags |
| **Rolling statistics** | 7-day and 30-day rolling mean and standard deviation |
| **Calendar features** | Day of year, month, week of year, is_weekend |
| **Fourier terms** | Sine/cosine transforms of day-of-year for seasonal cycles |
| **Trend features** | Cumulative sum decomposition for long-term drift |

---

## Pipeline Architecture

```
[Raw CSV: Historical Temperature Data]
              |
              v
   [Preprocessing]
   • Parse datetime index
   • Resample to daily frequency
   • IQR outlier detection & removal
   • Forward-fill missing values
              |
              v
   [Feature Engineering]
   • Lag features (T-1 to T-30)
   • Rolling mean/std (7d, 30d)
   • Fourier seasonal terms
   • Calendar features
              |
              v
   [Train / Test Split]
   • Chronological split (no data leakage)
              |
              v
   [XGBoost Regressor]
   • n_estimators, max_depth, learning_rate tuning
   • Early stopping on validation RMSE
              |
              v
   [Evaluation & Visualization]
   • RMSE, MAE, R²
   • Forecast vs. actual temperature plot
   • Feature importance bar chart
```

---

## Results

| Metric | Value |
|---|---|
| RMSE | **2.3°C** |
| MAE | ~1.8°C |
| R² | High (see notebook) |

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/tamer017/Cairo-Temperature-forecasting.git
cd Cairo-Temperature-forecasting

# Install dependencies
pip install xgboost pandas numpy matplotlib scikit-learn seaborn

# Run the pipeline
jupyter notebook Cairo_Temperature_Forecasting.ipynb
```

---

## Skills Demonstrated

- **Machine Learning:** XGBoost, gradient boosting, hyperparameter tuning, early stopping
- **Time-Series Engineering:** Lag features, rolling statistics, Fourier seasonal decomposition, chronological train/test split
- **Data Preprocessing:** IQR outlier detection, datetime indexing, resampling, missing value imputation
- **Visualization:** Matplotlib, Seaborn, forecast plots, feature importance charts
- **Python Stack:** pandas, NumPy, scikit-learn, XGBoost

---

## References

- [XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/)
- [Forecasting: Principles and Practice — Hyndman & Athanasopoulos](https://otexts.com/fpp3/)
