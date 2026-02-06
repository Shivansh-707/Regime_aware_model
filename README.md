📊 Regime-Aware Volatility Forecasting
Interactive Streamlit Demo

This project demonstrates how incorporating market regime detection improves volatility forecasting using deep learning.

We compare:

Model A → Standard LSTM

Model B → Regime-aware LSTM (HMM + Regime Embedding)

The model forecasts 5-day realized volatility for SPY using technical indicators and regime classification.

🚀 Live Demo

Deployed on Streamlit Cloud
(to be inserted)

🧠 Core Idea

Financial markets behave differently across regimes (low volatility, high volatility, crisis periods, etc.).

Instead of training a single model blindly, we:

Detect hidden market regimes using a Gaussian HMM

Embed regime information into an LSTM model

Compare performance against a standard LSTM

Result:

Improved RMSE

Improved MAE

Better stability during high-volatility regimes

🏗 Architecture
Data

15 years of SPY daily data

Log returns

Technical indicators:

RSI

MACD

ATR

Bollinger Bands

Rolling volatility

Moving averages

Volume ratio

Regime Detection

Gaussian Hidden Markov Model (3 states)

Features:

Returns

Rolling volatility

ATR

Volume ratio

Forecasting Model

Baseline Model

LSTM → Dense → Volatility prediction

Regime-Aware Model

LSTM → Regime Embedding → Concatenation → Dense → Prediction

📈 What the Dashboard Shows

Model comparison (RMSE & MAE)

Regime timeline visualization

Volatility distribution per regime

Actual vs predicted comparison

Error breakdown by regime

Current market regime inference

🛠 Tech Stack

Python

PyTorch

Streamlit

Scikit-learn

hmmlearn

yfinance

Matplotlib

Pandas / NumPy

📦 Project Structure
regime_aware_project/
│
├── app.py
├── model_A.pt
├── model_B.pt
├── scaler.pkl
├── scaler_regime.pkl
├── requirements.txt
└── README.md

▶️ Run Locally
pip install -r requirements.txt
python3 -m streamlit run app.py

📊 Why This Project Matters

Most retail volatility models ignore regime shifts.

This project demonstrates:

How regime conditioning improves robustness

How hidden states can enhance neural forecasting

A practical integration of probabilistic models + deep learning

🔬 Future Improvements

Transformer-based volatility model

Online learning

Regime transition probability analysis

Multi-asset extension

Live intraday mode

👨‍💻 Author

Shivansh Jha
Engineering Student | ML & Quant Research Enthusiast