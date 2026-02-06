import pandas as pd
import numpy as np
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
import warnings

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 30
LR = 0.001
SEQ_LEN = 30
NUM_REGIMES = 3

# 1️⃣ Download SPY data
ticker = yf.Ticker("SPY")
df = ticker.history(period="15y", interval="1d")
df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

# 2️⃣ Feature engineering
df['returns'] = np.log(df['Close'] / df['Close'].shift(1))
df['vol_5d'] = df['returns'].rolling(5).std() * np.sqrt(252)
df['vol_10d'] = df['returns'].rolling(10).std() * np.sqrt(252)
df['atr'] = (df['High'] - df['Low']).rolling(14).mean()
df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
df['target_vol'] = df['returns'].shift(-5).rolling(5).std() * np.sqrt(252)
df = df.dropna()

features = ['returns', 'vol_5d', 'vol_10d', 'atr', 'volume_ratio']

# 3️⃣ Regime detection
scaler_regime = StandardScaler()
regime_scaled = scaler_regime.fit_transform(df[features].values)

hmm = GaussianHMM(n_components=NUM_REGIMES, covariance_type="full", n_iter=100, random_state=42)
hmm.fit(regime_scaled)
df['regime'] = hmm.predict(regime_scaled)

# 4️⃣ Scale features for LSTM
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# 5️⃣ Sequence creation
def create_sequences(df, feature_cols, target_col, regime_col, seq_len=SEQ_LEN):
    X, y, r = [], [], []
    for i in range(seq_len, len(df)):
        X.append(df[feature_cols].iloc[i-seq_len:i].values)
        y.append(df[target_col].iloc[i])
        r.append(df[regime_col].iloc[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(r, dtype=np.int64)

X, y, regimes = create_sequences(df, features, 'target_vol', 'regime')

# Train/test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
regime_train, regime_test = regimes[:split], regimes[split:]

# 6️⃣ Dataset class
class VolDataset(Dataset):
    def __init__(self, X, y, regimes=None, use_regime=False):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        self.regimes = torch.from_numpy(regimes).long() if regimes is not None else None
        self.use_regime = use_regime
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.use_regime:
            return self.X[idx], self.regimes[idx], self.y[idx]
        return self.X[idx], self.y[idx]

# 7️⃣ LSTM model
class VolatilityLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_regimes=NUM_REGIMES, use_regime=False):
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

# 8️⃣ Training function
def train_model(model, dataloader, epochs=EPOCHS, lr=LR):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            if model.use_regime:
                Xb, rb, yb = batch
                pred = model(Xb.to(DEVICE), rb.to(DEVICE))
            else:
                Xb, yb = batch
                pred = model(Xb.to(DEVICE))
            loss = criterion(pred, yb.to(DEVICE))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # Optional: print per epoch
        # print(f"Epoch {epoch+1}/{epochs} — Loss: {total_loss/len(dataloader):.6f}")
    return model

# 9️⃣ Create dataloaders
train_loader_A = DataLoader(VolDataset(X_train, y_train, use_regime=False), batch_size=64, shuffle=True)
train_loader_B = DataLoader(VolDataset(X_train, y_train, regime_train, use_regime=True), batch_size=64, shuffle=True)

# 🔟 Initialize and train models
model_A = VolatilityLSTM(input_size=len(features), use_regime=False)
model_B = VolatilityLSTM(input_size=len(features), use_regime=True)

model_A = train_model(model_A, train_loader_A)
model_B = train_model(model_B, train_loader_B)

# 1️⃣1️⃣ Save models and scalers
torch.save(model_A.state_dict(), "model_A.pt")
torch.save(model_B.state_dict(), "model_B.pt")
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("scaler_regime.pkl", "wb") as f:
    pickle.dump(scaler_regime, f)
print("Saved model_A.pt, model_B.pt, scaler.pkl, scaler_regime.pkl")

# 1️⃣2️⃣ Evaluate
model_A.eval()
model_B.eval()
with torch.no_grad():
    X_test_tensor = torch.from_numpy(X_test).float().to(DEVICE)
    regime_test_tensor = torch.from_numpy(regime_test).long().to(DEVICE)
    predA = model_A(X_test_tensor).cpu().numpy()
    predB = model_B(X_test_tensor, regime_test_tensor).cpu().numpy()

def calc_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

rmse_A, mae_A, r2_A = calc_metrics(y_test, predA)
rmse_B, mae_B, r2_B = calc_metrics(y_test, predB)

print(f"\nModel A: RMSE {rmse_A:.6f} | MAE {mae_A:.6f} | R2 {r2_A:.6f}")
print(f"Model B: RMSE {rmse_B:.6f} | MAE {mae_B:.6f} | R2 {r2_B:.6f}")

# 1️⃣3️⃣ Improvement
print(f"\nImprovement — RMSE: {(rmse_A - rmse_B)/rmse_A*100:.2f}% | MAE: {(mae_A - mae_B)/mae_A*100:.2f}%")
