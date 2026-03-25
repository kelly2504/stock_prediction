import torch
import numpy
import requests
import pandas as pd

from dotenv import load_dotenv
import os 
from datetime import datetime
import time

import matplotlib.pyplot as plt
from torch import nn
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# By Kelly Wan - 02/02/2026
# This program predicts the stock of NVIDIA using an API call
## API KEYS IN .env file
load_dotenv()
#MARK: constants
learning_rate=0.01
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


# Convert bars to a pandas DataFrame
df = bars.df.reset_index()
df = df[df["symbol"]==SYMBOL]

print(df.head())

# Convert numeric columns (prefer OHLCV) to PyTorch tensors
preferred_cols = ['open', 'high', 'low', 'close', 'volume']

# #features we created
# df["return"] = df["close"].pct_change()
# df["volatility"] = df["return"].rolling(5).std()
# df["ma_5"] = df["close"].rolling(5).mean()
# df["ma_10"] = df["close"].rolling(10).mean()
# df["ma_20"] = df["close"].rolling(20).mean()

# # df["target"] = df["close"].shift(-1)
# # target = next log return
# df = df.dropna()

# features = [
#     "close",
#     "volume",
#     "volatility",
#     "ma_5",
#     "ma_10",
#     "ma_20",
# ]
close = df["close"]

prices = torch.from_numpy(close.astype('float32').values.copy())

returns = torch.log(prices[1:] / prices[:-1])
returns = (returns - returns.mean()) / returns.std()

# # cleaning up the data and setting tensors
# X = df[features]
# y = df["target"]

# #convert dataframes to numpy arrays, then to tensor
# X_tensor = torch.from_numpy(X.astype('float32').values.copy())
# y_tensor = torch.from_numpy(y.astype('float32').values.copy()).unsqueeze(dim=1)

# print(f"X shape: {X_tensor.shape}")
# print(f"y shape: {y_tensor.shape}")

# # set the weight and bias
# weight = 0.7
# bias = 0.3

def make_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return torch.stack(X), torch.stack(y)

X, y = make_sequences(returns, seq_len=30)


class learningModel(nn.Module):
    def __init__(self, input_features=6):
        super().__init__()
        self.linear = nn.Linear(input_features, 1)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)
        
# class StockLSTM(nn.Module):
#     def __init__(self, hidden=64):
#         super().__init__()
        
## THIS GOOD
# break the data into training and testing data
train_split = int(0.8 * len(X))

X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

print(len(X_train), len(y_train), len(X_test), len(y_test))


#MARK: Plot prediction
def plot_prediction(df, train_split, pred_returns):
    """
    Plots actual vs predicted close prices over time.
    
    df: full dataframe with ['timestamp', 'close']
    train_split: index where test set starts
    pred_returns: model-predicted returns (torch tensor)
    """
    
    #extract test portion
    test_df = df.iloc[train_split:].copy()
    test_df.reset_index(drop=True, inplace=True)
    
    #Last known close before test period
    last_train_close = df.iloc[train_split - 1]["close"]
    
    # Convert predicted returns to numpy
    pred_returns = pred_returns.squeeze().detach().cpu().numpy()
    
    # Reconstruct predicted close prices
    predicted_prices = []
    current_price = last_train_close
    
    for r in pred_returns:
        current_price *= float(torch.exp(torch.tensor(r)))
        predicted_prices.append(current_price)
    
    ret_mean = returns.mean()
    ret_std = returns.std()

    pred_returns = pred_returns * ret_std + ret_mean
    
    plt.figure(figsize=(12,6))
    plt.plot(test_df["timestamp"], test_df["close"], label="Actual")
    plt.plot(test_df["timestamp"], prices, label="Predicted")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Close Price")



model_1 = learningModel()

list(model_1.parameters())
model_1.state_dict()

with torch.inference_mode():
    y_preds = model_1(X_test)
    
# plot_prediction(prediction=y_preds)

loss_fn = nn.L1Loss()

optimizer = torch.optim.SGD(params=model_1.parameters(), lr=learning_rate)

train_loss_values = []
test_loss_values = []
epoch_count = []

# training loop
for e in range(epochs):
    model_1.train()
    y_pred = model_1(X_train)
    
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    
    loss.backward()
    optimizer.step()
    
    # evaluate
    model_1.eval()
    
    with torch.inference_mode():
        test_pred = model_1(X_test) #forward pass on the test data
        test_loss = loss_fn(test_pred, y_test.type(torch.float)) #calculates loss on test data

        #print out what's happening
        if e % 10 == 0:
            epoch_count.append(e)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            print(f"Epoch: {e} | MAE Train loss: {loss} | MAE Test loss: {test_loss}")


model_1.eval()
with torch.inference_mode():
    y_preds = model_1(X_test)
    
plot_prediction(
    df=df,
    train_split=train_split,
    pred_returns=y_preds
)

     
plt.show()