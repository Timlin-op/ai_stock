import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.layers import LSTM, Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping

df=yf.Ticker("BTC-USD").history(period="10y")

df=df.filter(["Close"])
df=df.rename(columns={"Close": "GT"})
print(df)

plt.style.use("seaborn-darkgrid")
plt.xlabel("Date")
plt.ylabel("Price")
plt.plot(df["GT"], linewidth=1)


scaler=MinMaxScaler(feature_range=(0,1))
scaled_prices=scaler.fit_transform(df.values)
print(scaled_prices)

MOVING_WIN_SIZE=60

all_x, all_y = [], []
for i in range(len(scaled_prices)-MOVING_WIN_SIZE):
    x=scaled_prices[i:i+MOVING_WIN_SIZE]
    y=scaled_prices[i+MOVING_WIN_SIZE]
    all_x.append(x)
    all_y.append(y)
all_x, all_y= np.array(all_x), np.array(all_y)
print(all_x.shape)
print(all_y.shape)

DS_SPLIT=0.8

train_ds_size=round(all_x.shape[0]*DS_SPLIT)
train_x,train_y=all_x[:train_ds_size],all_y[:train_ds_size]
test_x,test_y=all_x[train_ds_size:],all_y[train_ds_size:]
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

model= Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(train_x.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))
print(model.summary())

print(model.compile(optimizer="adam",loss="mean_squared_error"))

callback=EarlyStopping(monitor="val_loss",
                       patience=10,
                       restore_best_weights=True)
model.fit(train_x,train_y,
          validation_split=0.2,
          callbacks=[callback],
          epochs=1000)

preds= model.predict(test_x)
print(preds)

preds = scaler.inverse_transform(preds)
print(preds)

train_df=df[:train_ds_size+MOVING_WIN_SIZE]
test_df= df[train_ds_size+MOVING_WIN_SIZE:]
test_df=test_df.assign(Predict=preds)

plt.xlabel("Date")
plt.ylabel("Price")
plt.plot(train_df["GT"], linewidth=2)
plt.plot(test_df["GT"], linewidth=2)
plt.plot(test_df["Predict"], linewidth=1)
plt.legend(["Train","GT","Predict"])
print(plt.show())


