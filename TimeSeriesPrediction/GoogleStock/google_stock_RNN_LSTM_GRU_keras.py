#https://www.geeksforgeeks.org/time-series-forecasting-using-recurrent-neural-networks-rnn-in-tensorflow/

from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import tensorflow as tf
import matplotlib.pyplot as plt
import math


start_date = dt.datetime(2020, 4, 1)
end_date = dt.datetime(2023, 4, 1)

data = yf.download("GOOGL", start_date, end_date)

lenght_training_data = math.ceil(len(data)*0.8)
scaler = MinMaxScaler(feature_range=(0, 1))

train = data[:lenght_training_data].iloc[:, :1]
dataset_train = train.Open.values
dataset_train = np.reshape(dataset_train, (-1, 1))
scaled_train = scaler.fit_transform(dataset_train)

test = data[lenght_training_data:].iloc[:, :1]
dataset_test = test.Open.values
dataset_test = np.reshape(dataset_test, (-1, 1))
scaled_test = scaler.fit_transform(dataset_test)

X_train = []
y_train = []
for i in range(50, len(scaled_train)):
    X_train.append(scaled_train[i-50:i, 0])
    y_train.append(scaled_train[i, 0])

X_test = []
y_test = []
for i in range(50, len(scaled_test)):
    X_test.append(scaled_test[i-50:i, 0])
    y_test.append(scaled_test[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
y_train = np.reshape(y_train, (y_train.shape[0], 1))

X_test, y_test = np.array(X_test), np.array(y_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
y_test = np.reshape(y_test, (y_test.shape[0], 1))

rnn_regressor = tf.keras.Sequential()
rnn_regressor.add(tf.keras.layers.SimpleRNN(units=50, activation="tanh", return_sequences=True, input_shape=(X_train.shape[1], 1)))
rnn_regressor.add(tf.keras.layers.Dropout(0.2))
rnn_regressor.add(tf.keras.layers.SimpleRNN(units=50, activation="tanh", return_sequences=True))
rnn_regressor.add(tf.keras.layers.SimpleRNN(units=50, activation="tanh", return_sequences=True))
rnn_regressor.add(tf.keras.layers.SimpleRNN(units=50))
rnn_regressor.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

rnn_regressor.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True), loss="mean_squared_error")
rnn_regressor.fit(X_train, y_train, epochs=20, batch_size=2)

lstm_regressor = tf.keras.Sequential()
lstm_regressor.add(tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
lstm_regressor.add(tf.keras.layers.LSTM(50, return_sequences=False))
lstm_regressor.add(tf.keras.layers.Dense(25))
lstm_regressor.add(tf.keras.layers.Dense(1))

lstm_regressor.compile(optimizer='adam', loss='mean_squared_error', metrics=["accuracy"])
lstm_regressor.fit(X_train, y_train, batch_size=1, epochs=12)

gru_regressor = tf.keras.Sequential()
gru_regressor.add(tf.keras.layers.GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1), activation='tanh'))
gru_regressor.add(tf.keras.layers.Dropout(0.2))
gru_regressor.add(tf.keras.layers.GRU(units=50, return_sequences=True, activation='tanh'))
gru_regressor.add(tf.keras.layers.GRU(units=50, return_sequences=True, activation='tanh'))
gru_regressor.add(tf.keras.layers.GRU(units=50, activation='tanh'))
gru_regressor.add(tf.keras.layers.Dense(units=1, activation='relu'))
gru_regressor.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=False), loss='mean_squared_error')
gru_regressor.fit(X_train, y_train, epochs=20, batch_size=1)

y_RNN = rnn_regressor.predict(X_test)
y_RNN_O = scaler.inverse_transform(y_RNN)

y_LSTM = lstm_regressor.predict(X_test)
y_LSTM_O = scaler.inverse_transform(y_LSTM)

y_GRU = gru_regressor.predict(X_test)
y_GRU_O = scaler.inverse_transform(y_GRU)

fig, axs = plt.subplots(3, figsize=(18, 12), sharex=True, sharey=True)
fig.suptitle('Model Predictions')

# Plot for RNN predictions
axs[0].plot(train.index[150:], train.Open[150:], label="train_data", color="b")
axs[0].plot(test.index, test.Open, label="test_data", color="g")
axs[0].plot(test.index[50:], y_RNN_O, label="y_RNN", color="brown")
axs[0].legend()
axs[0].title.set_text("Basic RNN")

# Plot for LSTM predictions
axs[1].plot(train.index[150:], train.Open[150:], label="train_data", color="b")
axs[1].plot(test.index, test.Open, label="test_data", color="g")
axs[1].plot(test.index[50:], y_LSTM_O, label="y_LSTM", color="orange")
axs[1].legend()
axs[1].title.set_text("LSTM")

# Plot for GRU predictions
axs[2].plot(train.index[150:], train.Open[150:], label="train_data", color="b")
axs[2].plot(test.index, test.Open, label="test_data", color="g")
axs[2].plot(test.index[50:], y_GRU_O, label="y_GRU", color="red")
axs[2].legend()
axs[2].title.set_text("GRU")

plt.xlabel("Days")
plt.ylabel("Open price")

plt.show()