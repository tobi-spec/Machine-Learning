from sklearn.preprocessing import MinMaxScaler
import numpy as np
import yfinance as yf
import datetime as dt
import tensorflow as tf
import matplotlib.pyplot as plt
import math


class StockDataSet:
    def __init__(self, dataseries):
        self.dataseries = dataseries
        self.inputs = []
        self.targets = []

    def normalise(self):
        dataset_train = np.reshape(self.dataseries, (-1, 1))
        self.dataseries = scaler.fit_transform(dataset_train)
        return self

    def split_input_target(self):
        for i in range(50, len(self.dataseries)):
            self.inputs.append(self.dataseries[i - 50:i, 0])
            self.targets.append(self.dataseries[i, 0])
        return self

    def reshape(self):
        self.inputs = np.array(self.inputs)
        self.inputs = np.reshape(self.inputs, (self.inputs.shape[0], self.inputs.shape[1], 1))
        self.targets = np.array(self.targets)
        self.targets = np.reshape(self.targets, (self.targets.shape[0], 1))
        return self


start_date = dt.datetime(2020, 4, 1)
end_date = dt.datetime(2023, 4, 1)
data = yf.download("GOOGL", start_date, end_date)

lenght_training_data = math.ceil(len(data)*0.8)
scaler = MinMaxScaler(feature_range=(0, 1))

train = data[:lenght_training_data].iloc[:, :1]
dataset_train = train.loc[:, "Open"]
train_dataset = StockDataSet(dataset_train)
train_dataset.normalise().split_input_target().reshape()

test = data[lenght_training_data:].iloc[:, :1]
dataset_test = test.loc[:, "Open"]
test_dataset = StockDataSet(dataset_test)
test_dataset.normalise().split_input_target().reshape()


rnn_regressor = tf.keras.Sequential()
rnn_regressor.add(tf.keras.layers.SimpleRNN(units=50, activation="tanh", return_sequences=True, input_shape=(train_dataset.inputs.shape[1], 1)))
rnn_regressor.add(tf.keras.layers.Dropout(0.2))
rnn_regressor.add(tf.keras.layers.SimpleRNN(units=50, activation="tanh", return_sequences=True))
rnn_regressor.add(tf.keras.layers.SimpleRNN(units=50, activation="tanh", return_sequences=True))
rnn_regressor.add(tf.keras.layers.SimpleRNN(units=50))
rnn_regressor.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

rnn_regressor.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True), loss="mean_squared_error")
rnn_regressor.fit(train_dataset.inputs, train_dataset.targets, epochs=20, batch_size=2)

lstm_regressor = tf.keras.Sequential()
lstm_regressor.add(tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(train_dataset.inputs.shape[1], 1)))
lstm_regressor.add(tf.keras.layers.LSTM(50, return_sequences=False))
lstm_regressor.add(tf.keras.layers.Dense(25))
lstm_regressor.add(tf.keras.layers.Dense(1))

lstm_regressor.compile(optimizer='adam', loss='mean_squared_error', metrics=["accuracy"])
lstm_regressor.fit(train_dataset.inputs, train_dataset.targets, batch_size=1, epochs=12)

gru_regressor = tf.keras.Sequential()
gru_regressor.add(tf.keras.layers.GRU(units=50, return_sequences=True, input_shape=(train_dataset.inputs.shape[1], 1), activation='tanh'))
gru_regressor.add(tf.keras.layers.Dropout(0.2))
gru_regressor.add(tf.keras.layers.GRU(units=50, return_sequences=True, activation='tanh'))
gru_regressor.add(tf.keras.layers.GRU(units=50, return_sequences=True, activation='tanh'))
gru_regressor.add(tf.keras.layers.GRU(units=50, activation='tanh'))
gru_regressor.add(tf.keras.layers.Dense(units=1, activation='relu'))
gru_regressor.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=False), loss='mean_squared_error')
gru_regressor.fit(train_dataset.inputs, train_dataset.targets, epochs=10, batch_size=1)

y_RNN = rnn_regressor.predict(test_dataset.inputs)
y_RNN_O = scaler.inverse_transform(y_RNN)

y_LSTM = lstm_regressor.predict(test_dataset.inputs)
y_LSTM_O = scaler.inverse_transform(y_LSTM)

y_GRU = gru_regressor.predict(test_dataset.inputs)
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

plt.savefig("./comparison_RNN_LSTM_GRU_keras.png")
plt.show()