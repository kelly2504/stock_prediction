import torch
import torch.nn as nn
import numpy as np
import requests
import pandas as pd

from dotenv import load_dotenv
import os 
from datetime import datetime
import time

import matplotlib.pyplot as plt
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from sklearn.preprocessing import MinMaxScaler

load_dotenv()

#MARK: settings
learning_rate = 0.01
epochs = 100

SYMBOL = "NVDA"
START_DATE = datetime(2022, 1, 1)
END_DATE = datetime(2026, 1, 15)

Api_key = os.getenv("API_KEY")
Api_secret = os.getenv("API_SECRET")
client = StockHistoricalDataClient(Api_key,Api_secret)

request_params = StockBarsRequest(
    symbol_or_symbols=SYMBOL,
    timeframe=TimeFrame.Day,
    start=START_DATE,
    end=END_DATE,
)

bars = client.get_stock_bars(request_params)
df = bars.df.reset_index()
df = df[df['symbol'] == 'NVDA']
df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
df.set_index('timestamp', inplace=True)

print(df.head())

#prepping other features: returns, volatility and moving averages
df['return'] = df['close'].pct_change()
df['ma_20'] = df['close'].rolling(20).mean()
df['ma_50'] = df['close'].rolling(50).mean()
df['volatility'] = df['return'].rolling(20).std()

df.dropna(inplace=True)

#prepaing time-series dataset
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

sequence_length = 60  # 60 days history

X = []
y = []

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])
    y.append(scaled_data[i, 3])  # predicting close price

X = np.array(X)
y = np.array(y)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last timestep
        out = self.fc(out)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTMModel(input_size=X.shape[2], hidden_size=64, num_layers=2).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y, dtype=torch.float32).to(device).view(-1,1)

epochs = 100

for epoch in range(epochs):
    #put in training mode
    model.train()
    
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 5 == 0:
        print(f"Epoch[{epoch}/{epochs}], Loss: {loss.item():.6f}")

model.eval()

with torch.no_grad():
    predictions = model(X_tensor)

predictions = predictions.cpu().numpy()

#reverse scaling 
pred_full = np.zeros((len(predictions), scaled_data.shape[1]))
pred_full[:, 3] = predictions.flatten()

def plot_prediction():
    pred_prices = scaler.inverse_transform(pred_full)[:, 3]
    plt.figure(figsize=(12,6))
    plt.plot(df.index[sequence_length:], df['close'][sequence_length:], label="Actual")
    plt.plot(df.index[sequence_length:], pred_prices, label="Predicted")
    plt.legend()
    plt.show()
    
plot_prediction()
    
    