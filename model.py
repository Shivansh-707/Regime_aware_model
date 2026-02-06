import pandas as pd
import numpy as np
import yfinance as yf
import ta
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
<<<<<<< HEAD
import matplotlib.pyplot as plt
import warnings
import pickle

warnings.filterwarnings('ignore')

# Download data
ticker = yf.Ticker("SPY")
df = ticker.history(period="15y", interval="1d")
df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

# Compute returns and features
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

# Rolling volatility features
df['vol_5d'] = df['returns'].rolling(5).std() * np.sqrt(252)
df['vol_10d'] = df['returns'].rolling(10).std() * np.sqrt(252)
df['vol_20d'] = df['returns'].rolling(20).std() * np.sqrt(252)

# Target: 5-day realized volatility
df['target_vol'] = df['returns'].shift(-5).rolling(5).std() * np.sqrt(252)
df = df.dropna()

# Regime detection
regime_features = df[['returns', 'vol_5d', 'atr', 'volume_ratio']].values
scaler_regime = StandardScaler()
regime_scaled = scaler_regime.fit_transform(regime_features)

hmm = GaussianHMM(n_components=3, covariance_type="full", n_iter=100, random_state=42)
hmm.fit(regime_scaled)
df['regime'] = hmm.predict(regime_scaled)

# Prepare features for LSTM
feature_cols = ['returns', 'ma5', 'ma10', 'ma20', 'rsi', 'macd', 'macd_signal', 
                'atr', 'bb_high', 'bb_low', 'bb_mid', 'volume_ratio', 
                'vol_5d', 'vol_10d', 'vol_20d']
scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

# Create sequences
seq_len = 30

def create_sequences(data, features, target, regime):
    X, y, r = [], [], []
    for i in range(seq_len, len(data)):
        X.append(data[features].iloc[i-seq_len:i].values)
        y.append(data[target].iloc[i])
        r.append(data[regime].iloc[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(r, dtype=np.int64)

X, y, regimes = create_sequences(df, feature_cols, 'target_vol', 'regime')

# Train/test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
regime_train, regime_test = regimes[:split], regimes[split:]

# Dataset class
class VolDataset(Dataset):
    def __init__(self, X, y, regimes, use_regime):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        self.regimes = torch.from_numpy(regimes).long()
        self.use_regime = use_regime
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.use_regime:
            return self.X[idx], self.regimes[idx], self.y[idx]
        return self.X[idx], self.y[idx]

# LSTM Model
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

# Training function
def train_model(model, train_loader, epochs=20, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            if model.use_regime:
                X_batch, regime_batch, y_batch = batch
                pred = model(X_batch, regime_batch)
            else:
                X_batch, y_batch = batch
                pred = model(X_batch)
            
            loss = criterion(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    return model

# Model A: without regime
train_dataset_A = VolDataset(X_train, y_train, regime_train, use_regime=False)
train_loader_A = DataLoader(train_dataset_A, batch_size=64, shuffle=True)
model_A = VolatilityLSTM(input_size=len(feature_cols), use_regime=False)
model_A = train_model(model_A, train_loader_A, epochs=20)

# Model B: with regime
train_dataset_B = VolDataset(X_train, y_train, regime_train, use_regime=True)
train_loader_B = DataLoader(train_dataset_B, batch_size=64, shuffle=True)
model_B = VolatilityLSTM(input_size=len(feature_cols), use_regime=True)
model_B = train_model(model_B, train_loader_B, epochs=20)

# Save models and scalers
torch.save(model_A.state_dict(), "model_A.pt")
torch.save(model_B.state_dict(), "model_B.pt")
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("scaler_regime.pkl", "wb") as f:
    pickle.dump(scaler_regime, f)
print("Saved model_A.pt, model_B.pt, scaler.pkl, scaler_regime.pkl")

# Predictions
model_A.eval()
model_B.eval()
with torch.no_grad():
    X_test_tensor = torch.from_numpy(X_test).float()
    regime_test_tensor = torch.from_numpy(regime_test).long()
    pred_A = model_A(X_test_tensor).cpu().numpy()
    pred_B = model_B(X_test_tensor, regime_test_tensor).cpu().numpy()

# Metrics
def calc_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

rmse_A, mae_A, r2_A = calc_metrics(y_test, pred_A)
rmse_B, mae_B, r2_B = calc_metrics(y_test, pred_B)

print(f"\nModel A (No Regime):")
print(f"RMSE: {rmse_A:.4f} | MAE: {mae_A:.4f} | R²: {r2_A:.4f}")
print(f"\nModel B (With Regime):")
print(f"RMSE: {rmse_B:.4f} | MAE: {mae_B:.4f} | R²: {r2_B:.4f}")

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Regime timeline
test_dates = df.index[split+seq_len:]
axes[0, 0].scatter(test_dates, regime_test, c=regime_test, cmap='viridis', s=1)
axes[0, 0].set_title('Market Regimes Over Time')
axes[0, 0].set_ylabel('Regime')

# Volatility by regime
for r in range(3):
    mask = regime_test == r
    axes[0, 1].hist(y_test[mask], alpha=0.5, bins=30, label=f'Regime {r}')
axes[0, 1].set_title('Volatility Distribution by Regime')
axes[0, 1].legend()

# Predictions comparison
axes[1, 0].plot(y_test[:200], label='Actual', alpha=0.7)
axes[1, 0].plot(pred_A[:200], label='Model A', alpha=0.7)
axes[1, 0].plot(pred_B[:200], label='Model B', alpha=0.7)
axes[1, 0].set_title('Actual vs Predicted Volatility')
axes[1, 0].legend()

# Errors by regime
errors_B = np.abs(y_test - pred_B)
regime_errors = [errors_B[regime_test == r].mean() for r in range(3)]
axes[1, 1].bar(range(3), regime_errors, color=['blue', 'orange', 'green'])
axes[1, 1].set_title('Mean Absolute Error by Regime (Model B)')
axes[1, 1].set_xlabel('Regime')
axes[1, 1].set_ylabel('MAE')

plt.tight_layout()
plt.savefig('regime_volatility_analysis.png', dpi=300)
plt.show()
=======
import pickle
import warnings

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

############################################
# 1️⃣ Download Data
############################################

ticker = yf.Ticker("SPY")
df = ticker.history(period="15y", interval="1d")
df = df[['Open','High','Low','Close','Volume']].dropna()

df['returns'] = np.log(df['Close'] / df['Close'].shift(1))
df['vol_5']  = df['returns'].rolling(5).std() * np.sqrt(252)
df['vol_10'] = df['returns'].rolling(10).std() * np.sqrt(252)
df['vol_20'] = df['returns'].rolling(20).std() * np.sqrt(252)
df['rsi'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
df['atr'] = ta.volatility.AverageTrueRange(
    df['High'], df['Low'], df['Close'], window=14
).average_true_range()

df['target'] = df['returns'].shift(-5).rolling(5).std() * np.sqrt(252)
df = df.dropna()

############################################
# 2️⃣ Regime Detection (HMM)
############################################

regime_features = df[['returns','vol_5','atr']].values
scaler_regime = StandardScaler()
regime_scaled = scaler_regime.fit_transform(regime_features)

hmm = GaussianHMM(
    n_components=3,
    covariance_type="full",
    n_iter=200,
    random_state=42
)
hmm.fit(regime_scaled)
df['regime'] = hmm.predict(regime_scaled)

############################################
# 3️⃣ Feature Scaling
############################################

feature_cols = ['returns','vol_5','vol_10','vol_20','rsi','atr']
scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

############################################
# 4️⃣ Sequence Creation
############################################

SEQ_LEN = 30

def create_sequences(data):
    X, y, r = [], [], []
    for i in range(SEQ_LEN, len(data)):
        X.append(data[feature_cols].iloc[i-SEQ_LEN:i].values)
        y.append(data['target'].iloc[i])
        r.append(data['regime'].iloc[i])
    return np.array(X, dtype=np.float32), \
           np.array(y, dtype=np.float32), \
           np.array(r, dtype=np.int64)

X, y, regimes = create_sequences(df)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
reg_train, reg_test = regimes[:split], regimes[split:]

############################################
# 5️⃣ Dataset
############################################

class VolDataset(Dataset):
    def __init__(self, X, y, regimes, use_regime):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
        self.regimes = torch.tensor(regimes)
        self.use_regime = use_regime

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.use_regime:
            return self.X[idx], self.regimes[idx], self.y[idx]
        else:
            return self.X[idx], self.y[idx]

############################################
# 6️⃣ Model
############################################

class RegimeLSTM(nn.Module):
    def __init__(self, input_size, hidden=64, use_regime=False):
        super().__init__()
        self.use_regime = use_regime
        self.lstm = nn.LSTM(input_size, hidden, batch_first=True)

        if use_regime:
            self.embed = nn.Embedding(3, 8)
            self.fc = nn.Linear(hidden + 8, 1)
        else:
            self.fc = nn.Linear(hidden, 1)

    def forward(self, x, regime=None):
        out, _ = self.lstm(x)
        last = out[:, -1, :]

        if self.use_regime:
            emb = self.embed(regime)
            last = torch.cat([last, emb], dim=1)

        return self.fc(last).squeeze(-1)  # SAFE squeeze

############################################
# 7️⃣ Training
############################################

def train(model, loader, epochs=25):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        model.train()
        for batch in loader:
            optimizer.zero_grad()

            if model.use_regime:
                xb, rb, yb = batch
                xb, rb, yb = xb.to(DEVICE), rb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb, rb)
            else:
                xb, yb = batch
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)

            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

    return model

############################################
# 8️⃣ Train Models
############################################

trainA = VolDataset(X_train, y_train, reg_train, False)
trainB = VolDataset(X_train, y_train, reg_train, True)

loaderA = DataLoader(trainA, batch_size=64, shuffle=True)
loaderB = DataLoader(trainB, batch_size=64, shuffle=True)

modelA = train(RegimeLSTM(len(feature_cols), use_regime=False), loaderA)
modelB = train(RegimeLSTM(len(feature_cols), use_regime=True), loaderB)

############################################
# 9️⃣ Evaluation
############################################

modelA.eval()
modelB.eval()

with torch.no_grad():
    XA = torch.tensor(X_test).to(DEVICE)
    RB = torch.tensor(reg_test).to(DEVICE)

    predA = modelA(XA).cpu().numpy()
    predB = modelB(XA, RB).cpu().numpy()

def metrics(y_true, y_pred):
    return (
        np.sqrt(mean_squared_error(y_true, y_pred)),
        mean_absolute_error(y_true, y_pred),
        r2_score(y_true, y_pred)
    )

rmseA, maeA, r2A = metrics(y_test, predA)
rmseB, maeB, r2B = metrics(y_test, predB)

print("\nModel A:")
print(f"RMSE: {rmseA:.6f} | MAE: {maeA:.6f} | R2: {r2A:.6f}")

print("\nModel B:")
print(f"RMSE: {rmseB:.6f} | MAE: {maeB:.6f} | R2: {r2B:.6f}")

print("\nImprovement:")
print(f"RMSE: {(rmseA - rmseB)/rmseA * 100:.2f}%")
print(f"MAE:  {(maeA - maeB)/maeA * 100:.2f}%")

############################################
# 🔟 Save Artifacts
############################################

torch.save(modelA.state_dict(), "model_A.pt")
torch.save(modelB.state_dict(), "model_B.pt")

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("scaler_regime.pkl", "wb") as f:
    pickle.dump(scaler_regime, f)

print("\nSaved model_A.pt, model_B.pt, scaler.pkl, scaler_regime.pkl")
>>>>>>> f3d1bd7 (Final regime-aware update with improved MAE/RMSE and Streamlit dashboard)
