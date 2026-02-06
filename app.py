import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import torch
import pickle
from model import VolatilityLSTM, SEQ_LEN, DEVICE, features, NUM_REGIMES
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt

# ----------------------------
# 1️⃣ Load models and scalers
# ----------------------------
@st.cache_resource
def load_models():
    modelA = VolatilityLSTM(input_size=len(features), use_regime=False)
    modelA.load_state_dict(torch.load("model_A.pt", map_location=DEVICE))
    modelA.eval()

    modelB = VolatilityLSTM(input_size=len(features), use_regime=True)
    modelB.load_state_dict(torch.load("model_B.pt", map_location=DEVICE))
    modelB.eval()

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("scaler_regime.pkl", "rb") as f:
        scaler_regime = pickle.load(f)
    
    return modelA, modelB, scaler, scaler_regime

modelA, modelB, scaler, scaler_regime = load_models()

# ----------------------------
# 2️⃣ Streamlit UI layout
# ----------------------------
st.set_page_config(layout="wide")
st.title("Regime-Aware Volatility Forecasting")
st.markdown("Predict 5-day S&P 500 volatility using LSTMs with regime embeddings.")

# Sidebar for settings
st.sidebar.header("Settings")
period = st.sidebar.selectbox("Historical Data Period", ["1y","3y","5y","10y","15y"], index=4)

# ----------------------------
# 3️⃣ Download data
# ----------------------------
ticker = yf.Ticker("SPY")
df = ticker.history(period=period)
df = df[['Open','High','Low','Close','Volume']].dropna()

# ----------------------------
# 4️⃣ Feature engineering
# ----------------------------
df['returns'] = np.log(df['Close']/df['Close'].shift(1))
df['vol_5d'] = df['returns'].rolling(5).std() * np.sqrt(252)
df['vol_10d'] = df['returns'].rolling(10).std() * np.sqrt(252)
df['atr'] = (df['High'] - df['Low']).rolling(14).mean()
df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
df = df.dropna()

# ----------------------------
# 5️⃣ Regime detection
# ----------------------------
regime_scaled = scaler_regime.transform(df[features].values)
hmm = GaussianHMM(n_components=NUM_REGIMES, covariance_type="full", n_iter=100, random_state=42)
hmm.fit(regime_scaled)
df['regime'] = hmm.predict(regime_scaled)

# Latest regime for prediction
latest_regime = torch.tensor([df['regime'].iloc[-1]], dtype=torch.long).to(DEVICE)

# ----------------------------
# 6️⃣ Prepare sequence for prediction
# ----------------------------
X_seq = df[features].iloc[-SEQ_LEN:]
X_seq_scaled = scaler.transform(X_seq)
X_seq_tensor = torch.tensor(X_seq_scaled[np.newaxis, :, :], dtype=torch.float32).to(DEVICE)

# ----------------------------
# 7️⃣ Predictions
# ----------------------------
with torch.no_grad():
    predA = modelA(X_seq_tensor).cpu().item()
    predB = modelB(X_seq_tensor, latest_regime).cpu().item()

# ----------------------------
# 8️⃣ Display latest prediction
# ----------------------------
st.subheader("Latest 5-Day Volatility Prediction")
st.write(f"**Model A (No Regime):** {predA:.5f}")
st.write(f"**Model B (Regime-Aware):** {predB:.5f}")
st.write(f"**Improvement (MAE proxy):** {100*(predA - predB)/predA:.2f}%")
regime_dict = {0:"Low / Calm",1:"Medium",2:"High / Volatile"}
st.write(f"**Current Market Regime:** {regime_dict.get(latest_regime.item(),'Unknown')}")

# ----------------------------
# 9️⃣ Actionable suggestion
# ----------------------------
def action_suggestion(vol, regime):
    if regime == 0:
        return "Low volatility — normal position sizing is fine."
    elif regime == 1:
        return "Medium volatility — reduce positions or add hedges."
    else:
        return "High volatility — defensive positions and hedges recommended."

st.write(f"**Actionable Suggestion:** {action_suggestion(predB, latest_regime.item())}")

# ----------------------------
# 10️⃣ Plots
# ----------------------------
fig, axes = plt.subplots(2,2, figsize=(15,10))

# Volatility over time
axes[0,0].plot(df['Close'].index[-200:], df['vol_5d'].iloc[-200:], label='Actual Volatility')
axes[0,0].plot(df['Close'].index[-SEQ_LEN:], [predA]*SEQ_LEN, label='Model A Prediction')
axes[0,0].plot(df['Close'].index[-SEQ_LEN:], [predB]*SEQ_LEN, label='Model B Prediction')
axes[0,0].set_title("Volatility Prediction vs Actual")
axes[0,0].legend()

# Regime timeline
axes[0,1].scatter(df.index[-200:], df['regime'].iloc[-200:], c=df['regime'].iloc[-200:], cmap='viridis', s=5)
axes[0,1].set_title("Market Regimes Timeline")

# Volatility histogram by regime
for r in range(NUM_REGIMES):
    mask = df['regime']==r
    axes[1,0].hist(df['vol_5d'][mask], alpha=0.5, bins=30, label=f"Regime {r}")
axes[1,0].set_title("Volatility Distribution by Regime")
axes[1,0].legend()

# Mean absolute error by regime (approx using last prediction as proxy)
axes[1,1].bar(range(NUM_REGIMES), [abs(predB - df['vol_5d'][df['regime']==r].mean()) for r in range(NUM_REGIMES)],
             color=['blue','orange','green'])
axes[1,1].set_title("MAE by Regime (Model B)")
axes[1,1].set_xlabel("Regime")
axes[1,1].set_ylabel("MAE Proxy")

plt.tight_layout()
st.pyplot(fig)
