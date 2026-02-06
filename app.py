# app.py - polished Streamlit dashboard for regime-aware volatility models
# Place this file next to: model_A.pt, model_B.pt, scaler.pkl, scaler_regime.pkl
# Run locally with: python3 -m streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
import io
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Regime-aware Volatility — Demo", layout="wide")

# -------------------- User-tweakable params --------------------
SEQ_LEN = 30
FEATURE_COLS = ['returns', 'ma5', 'ma10', 'ma20', 'rsi', 'macd', 'macd_signal',
                'atr', 'bb_high', 'bb_low', 'bb_mid', 'volume_ratio',
                'vol_5d', 'vol_10d', 'vol_20d']
REGIME_NAMES = {0: 'Calm / Low vol', 1: 'Transitional / Medium vol', 2: 'Risk-off / High vol'}

# -------------------- Small helper classes (same as training) --------------------
class VolDataset(Dataset):
    def __init__(self, X, y, regimes, use_regime=False):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.regimes = torch.LongTensor(regimes)
        self.use_regime = use_regime
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        if self.use_regime:
            return self.X[idx], self.regimes[idx], self.y[idx]
        return self.X[idx], self.y[idx]

class VolatilityLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_regimes=3, use_regime=False):
        super().__init__()
        self.use_regime = use_regime
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        if use_regime:
            self.regime_embed = nn.Embedding(num_regimes, 8)
            self.fc = nn.Linear(hidden_size + 8, 1)
        else:
            self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x, regime=None):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        if self.use_regime:
            regime_emb = self.regime_embed(regime)
            combined = torch.cat([last_hidden, regime_emb], dim=1)
            return self.fc(combined).squeeze()
        return self.fc(last_hidden).squeeze()

# -------------------- Data preparation --------------------
@st.cache_data(show_spinner=False)
def download_and_prepare():
    ticker = yf.Ticker("SPY")
    df = ticker.history(period="15y", interval="1d")
    df = df[['Open','High','Low','Close','Volume']].dropna()
    df['returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['ma5'] = df['Close'].rolling(5).mean()
    df['ma10'] = df['Close'].rolling(10).mean()
    df['ma20'] = df['Close'].rolling(20).mean()
    df['rsi'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    macd = ta.trend.MACD(df['Close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
    bb = ta.volatility.BollingerBands(df['Close'])
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['bb_mid'] = bb.bollinger_mavg()
    df['volume_sma'] = df['Volume'].rolling(20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_sma']
    df['vol_5d'] = df['returns'].rolling(5).std() * np.sqrt(252)
    df['vol_10d'] = df['returns'].rolling(10).std() * np.sqrt(252)
    df['vol_20d'] = df['returns'].rolling(20).std() * np.sqrt(252)
    df['target_vol'] = df['returns'].shift(-5).rolling(5).std() * np.sqrt(252)
    df = df.dropna()
    return df

@st.cache_data(show_spinner=False)
def detect_regimes(df, _scaler_regime=None):
    features = df[['returns','vol_5d','atr','volume_ratio']].values
    if _scaler_regime is None:
        _scaler_regime = StandardScaler()
        scaled = _scaler_regime.fit_transform(features)
    else:
        scaled = _scaler_regime.transform(features)
    hmm = GaussianHMM(n_components=3, covariance_type='full', n_iter=100, random_state=42)
    hmm.fit(scaled)
    regimes = hmm.predict(scaled)
    return regimes, _scaler_regime


def create_sequences(df, seq_len=SEQ_LEN):
    X, y, r = [], [], []
    for i in range(seq_len, len(df)):
        X.append(df[FEATURE_COLS].iloc[i-seq_len:i].values)
        y.append(df['target_vol'].iloc[i])
        r.append(df['regime'].iloc[i])
    return np.array(X), np.array(y), np.array(r)

# -------------------- Utils --------------------

def load_pickle(fname):
    try:
        with open(fname, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None


def load_model_state(path, use_regime=False, device='cpu'):
    if not os.path.exists(path):
        return None
    model = VolatilityLSTM(input_size=len(FEATURE_COLS), use_regime=use_regime)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def calc_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def interpret_action(latest_pred, historical_preds, regime):
    # Decide thresholds from historical percentiles
    p33, p66 = np.percentile(historical_preds, [33,66])
    if latest_pred >= p66:
        advice = "High 5-day realized volatility expected — consider reducing directional exposure, tighten stops, or hedge."
        risk = "High"
    elif latest_pred >= p33:
        advice = "Medium volatility — reduce position sizing or add defensive hedges if overweight equities."
        risk = "Medium"
    else:
        advice = "Low short-term volatility expected — opportunities for carry/vol strategies, but watch regime shifts."
        risk = "Low"
    regime_text = REGIME_NAMES.get(int(regime), 'Unknown')
    return risk, advice, regime_text

# -------------------- UI --------------------
st.title("Regime-aware Volatility — Interactive Demo")
st.markdown("Small demo that compares a standard LSTM vs a regime-aware LSTM for 5-day realized volatility forecasting.")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    show_raw = st.checkbox("Show source data (head)", value=False)
    compare_mode = st.radio("Comparison mode", ["Side-by-side", "Single Model (choose)"])
    default_model = st.selectbox("If single model: select", ["Model A (No Regime)", "Model B (With Regime)"])
    lookback = st.slider("Plot lookback (days)", min_value=100, max_value=2000, value=600, step=50)
    run_inference = st.button("Run inference / refresh data")

# Load data
with st.spinner("Downloading data and preparing features (cached)..."):
    df = download_and_prepare()

if show_raw:
    st.dataframe(df.head())

# Detect regimes
with st.spinner("Detecting regimes..."):
    scaler_regime_saved = load_pickle('scaler_regime.pkl')
    regimes, scaler_regime = detect_regimes(df, scaler_regime_saved)
    df['regime'] = regimes

# Scale features using saved scaler if available
scaler_saved = load_pickle('scaler.pkl')
if scaler_saved is not None:
    df[FEATURE_COLS] = scaler_saved.transform(df[FEATURE_COLS])
else:
    scaler = StandardScaler()
    df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])

# Prepare sequences
X, y, regime_arr = create_sequences(df)
split = int(0.8 * len(X))
X_test = X[split:]
y_test = y[split:]
regime_test = regime_arr[split:]

# Load models
modelA = load_model_state('model_A.pt', use_regime=False)
modelB = load_model_state('model_B.pt', use_regime=True)

if (modelA is None) and (modelB is None):
    st.warning("No saved model files found (model_A.pt, model_B.pt). Please run training locally and place the saved files in this folder. You can still explore data and regimes.")

# Run inference when requested
if run_inference:
    device = 'cpu'
    preds_A = preds_B = None
    if modelA is not None:
        with torch.no_grad():
            preds_A = modelA(torch.FloatTensor(X_test)).numpy()
    if modelB is not None:
        with torch.no_grad():
            preds_B = modelB(torch.FloatTensor(X_test), torch.LongTensor(regime_test)).numpy()

    # Compute metrics
    metrics = {}
    if preds_A is not None:
        metrics['A'] = calc_metrics(y_test, preds_A)
    if preds_B is not None:
        metrics['B'] = calc_metrics(y_test, preds_B)

    # Build comparison table
    rows = []
    if 'A' in metrics:
        rows.append(['Model A (No Regime)', metrics['A'][0], metrics['A'][1], metrics['A'][2]])
    if 'B' in metrics:
        rows.append(['Model B (With Regime)', metrics['B'][0], metrics['B'][1], metrics['B'][2]])
    comp_df = pd.DataFrame(rows, columns=['Model', 'RMSE', 'MAE', 'R2'])

    st.subheader("Model comparison")
    col1, col2 = st.columns([2,1])
    with col1:
        st.dataframe(comp_df.style.format({ 'RMSE': '{:.5f}', 'MAE': '{:.5f}', 'R2': '{:.4f}' }))

    # Show improvement deltas if both present
    if ('A' in metrics) and ('B' in metrics):
        rmse_rel = (metrics['A'][0] - metrics['B'][0]) / metrics['A'][0] * 100
        mae_rel = (metrics['A'][1] - metrics['B'][1]) / metrics['A'][1] * 100
        with col2:
            st.metric("RMSE improvement (B vs A)", f"{rmse_rel:.2f}%")
            st.metric("MAE improvement (B vs A)", f"{mae_rel:.2f}%")

    # Visualizations - improved 2x2 layout
    st.subheader("Full visualization - choose which model predictions to inspect")
    inspect_choice = 'B' if (preds_B is not None) else 'A'
    if compare_mode == 'Single Model (choose)':
        inspect_choice = 'A' if default_model.startswith('Model A') else 'B'

    preds_inspect = preds_B if inspect_choice == 'B' else preds_A
    if preds_inspect is None:
        st.info("Selected model predictions not available. Toggle comparison mode or load the other model.")
    else:
        # limit lookback
        visible_idx = max(0, len(y_test) - lookback)
        dates = df.index[split + SEQ_LEN:][visible_idx:]
        y_vis = y_test[visible_idx:]
        p_vis = preds_inspect[visible_idx:]
        r_vis = regime_test[visible_idx:]

        fig, axes = plt.subplots(2,2, figsize=(14,10))
        # Regime timeline
        axes[0,0].scatter(dates, r_vis, c=r_vis, s=6)
        axes[0,0].set_title('Market Regimes (visible window)')
        axes[0,0].set_yticks([0,1,2])
        axes[0,0].set_yticklabels([REGIME_NAMES[i] for i in [0,1,2]])

        # Volatility distribution by regime
        for r in [0,1,2]:
            mask = r_vis == r
            axes[0,1].hist(y_vis[mask], alpha=0.5, bins=30, label=f'Regime {r}')
        axes[0,1].set_title('Volatility distribution by regime (visible)')
        axes[0,1].legend()

        # Predictions vs actual (zoomed)
        axes[1,0].plot(dates[:300], y_vis[:300], label='Actual', alpha=0.8)
        axes[1,0].plot(dates[:300], p_vis[:300], label=f'Predicted ({inspect_choice})', alpha=0.8)
        axes[1,0].set_title('Actual vs Predicted (first 300 points in window)')
        axes[1,0].legend()

        # Errors by regime
        errors = np.abs(y_vis - p_vis)
        regime_errors = [errors[r_vis == r].mean() if np.any(r_vis==r) else np.nan for r in [0,1,2]]
        axes[1,1].bar([0,1,2], regime_errors)
        axes[1,1].set_title('Mean Absolute Error by Regime (visible window)')
        axes[1,1].set_xticks([0,1,2])
        axes[1,1].set_xticklabels([REGIME_NAMES[i] for i in [0,1,2]])

        plt.tight_layout()
        st.pyplot(fig)

        # Latest insight + action
        latest_pred = float(p_vis[-1])
        latest_regime = int(r_vis[-1])
        risk, advice, regime_text = interpret_action(latest_pred, p_vis, latest_regime)

        st.markdown("---")
        colA, colB, colC = st.columns([2,3,2])
        with colA:
            st.subheader("Latest prediction")
            st.metric("Predicted 5-day volatility", f"{latest_pred:.5f}")
            st.write(f"Regime: **{regime_text}**")
        with colB:
            st.subheader("Actionable suggestion")
            if risk == 'High':
                st.error(advice)
            elif risk == 'Medium':
                st.warning(advice)
            else:
                st.success(advice)
        with colC:
            st.subheader("Quick take")
            st.write(f"Model: **{ 'Model B' if inspect_choice=='B' else 'Model A' }**")
            if ('A' in metrics) and ('B' in metrics):
                better = 'B' if metrics['B'][1] < metrics['A'][1] else 'A'
                st.write(f"Better MAE: **Model {better}**")

    # Offer CSV download of predictions & errors for further inspection
    if preds_A is not None or preds_B is not None:
        out_df = pd.DataFrame({
            'date': df.index[split+SEQ_LEN:],
            'actual': y_test,
            'regime': regime_test
        })
        if preds_A is not None:
            out_df['pred_A'] = preds_A
        if preds_B is not None:
            out_df['pred_B'] = preds_B
        csv_buf = out_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download predictions CSV", data=csv_buf, file_name='predictions.csv', mime='text/csv')

# Footer / next steps
st.markdown("---")
<<<<<<< HEAD
st.write("**Next steps / notes:**")
st.write("1. For production, avoid training on the web server — train locally and push model files (model_A.pt, model_B.pt, scaler.pkl, scaler_regime.pkl).")
st.write("2. If model files exceed GitHub limits (>50 MB), use Git LFS or host weights on a cloud bucket and download at runtime.")
st.write("3. Want dashboard styling tweaks, custom CSS, or exportable PNGs? Say the word and I'll add them.")
=======
st.write("I started my journey into the financial markets just like every other kid does, by starting a demat account to fill IPOs")
st.write("With my background in Mathematics and Computer Science, I quickly gravitated towards the quantitative side of things, and have been building models and strategies ever since")
st.write("This project is a small demo of how regime-aware models can provide better insights and forecasts in financial markets, which are inherently non-stationary and regime-switching")
st.write("If you're interested in learning more or collaborating, feel free to reach out on lainnovic11@gmail.com")

>>>>>>> f3d1bd7 (Final regime-aware update with improved MAE/RMSE and Streamlit dashboard)


