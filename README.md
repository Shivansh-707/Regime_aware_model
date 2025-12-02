Volatility Prediction with Regime-Aware LSTM
This project implements models to predict 5-day realized volatility of the SPY ETF using historical price data, technical indicators, and market regimes identified by a Gaussian Hidden Markov Model (HMM). Two LSTM-based models are trained and compared: one incorporating regime embeddings and one without regime information.

Overview
Data: 15 years of daily SPY price and volume data downloaded via Yahoo Finance.

Features: Returns, moving averages, RSI, MACD, ATR, Bollinger Bands, volume ratios, and rolling volatility.

Target: 5-day future realized volatility derived from log returns.

Regime Detection: 3 market regimes identified using Gaussian HMM on volatility and volume features.

Models: LSTM model without regime input (Model A) and LSTM model with regime embeddings (Model B).

Evaluation: RMSE, MAE, and R² metrics measure prediction accuracy.

Model Performance
Model A (No Regime):

RMSE: 0.1284

MAE: 0.0964

R²: -1.0428

Model B (With Regime):

RMSE: 0.1165

MAE: 0.0871

R²: -0.6816

Model B incorporating regime information outperforms Model A on all metrics, demonstrating the value of regime-aware modeling.

