# Predictive Maintenance System

This project predicts machine failure using time-series sensor data.

## Features
- RUL calculation
- Feature engineering (rolling, lag)
- Random Forest & XGBoost models
- SHAP explainability
- Flask API deployment

## Results
- Random Forest PR-AUC: 0.94
- XGBoost PR-AUC: 0.95

## Deployment
Flask API endpoint:
`/predict`

## How to Run
1. Install requirements
2. Run `app.py`
