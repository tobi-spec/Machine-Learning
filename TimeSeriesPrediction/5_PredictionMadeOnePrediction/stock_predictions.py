import pandas as pd
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
        for i in range(1, len(self.dataseries)):
            self.inputs.append(self.dataseries[i - 1:i, 0])
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


model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(50, return_sequences=True))
model.add(tf.keras.layers.LSTM(50, return_sequences=False))
model.add(tf.keras.layers.Dense(25))
model.add(tf.keras.layers.Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(train_dataset.inputs, train_dataset.targets, batch_size=1, epochs=6)

prediction_on_real_data = model.predict(test_dataset.inputs)

single_prediction_1 = model.predict([test_dataset.inputs[0]])
single_prediction_2 = model.predict([single_prediction_1])
print(test_dataset.inputs[0])
print(prediction_on_real_data[0])
print(single_prediction_1)
print(test_dataset.inputs[1])
print(prediction_on_real_data[0])
print(single_prediction_2)


results = pd.DataFrame()
results["targets"] = test_dataset.targets.flatten()
results["prediction_on_real_data"] = prediction_on_real_data
results.to_csv("./results.csv")

plt.plot(test_dataset.targets, label="test_data", color="blue")
plt.plot(prediction_on_real_data, label="prediction_on_real_data")
plt.xlabel("time")
plt.ylabel("Stock value")
plt.legend(loc="upper right")
plt.savefig("./test.png")
plt.show()