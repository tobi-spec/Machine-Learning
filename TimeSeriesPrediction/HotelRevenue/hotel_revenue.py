import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt

'''
Average Daily Rate (ADR) is recognised as one of the most important metrics for hotels.
Essentially, ADR is measuring the average price of a hotel room over a given period.
It is calculated as follows: ADR = Revenue ÷ sold rooms
'''


class HotelRevenueData:
    def __init__(self):
        self.data = pd.read_csv(filepath_or_buffer="./H1.csv",
                                parse_dates=[[2, 3, 5]],
                                )
        self.data.rename(columns={"ArrivalDateYear_ArrivalDArrivalDateMonth_ArrivalDateDayOfMonth": "Date"},
                         inplace=True)

    def get_ADR(self):
        return self.data.loc[:, "ADR"]

    def get_weekly_ADR(self):
        return self.data.groupby([pd.Grouper(key='Date', freq='W')])["ADR"].sum()


hotel_revenue = HotelRevenueData()
ADR_data = hotel_revenue.get_weekly_ADR().reset_index(drop=True)


def create_train_validation(data):
    train_size = int(len(data)*0.8)
    train = data.loc[:train_size]
    validation = data.loc[train_size:]
    return pd.DataFrame(train), pd.DataFrame(validation)


train, validation = create_train_validation(ADR_data)
scaler = MinMaxScaler(feature_range=(0, 1))
train = scaler.fit_transform(train)
validation = scaler.fit_transform(validation)


def create_timeseries(data, previous=1):
    inputs, targets = list(), list()
    for i in range(len(data)-previous-1):
        a = data[i: (i+previous), 0]
        inputs.append(a)
        targets.append(data[i + previous, 0])
    return np.array(inputs), np.array(targets)


lookback = 5
inputs_train, targets_train = create_timeseries(train, lookback)
inputs_validate, targets_validate = create_timeseries(validation, lookback)

inputs_train = np.reshape(inputs_train, (inputs_train.shape[0], 1, inputs_train.shape[1]))
inputs_validate = np.reshape(inputs_validate, (inputs_validate.shape[0], 1, inputs_validate.shape[1]))


def create_LSTM_model(inputs, targets, lookback):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(4, input_shape=(1, lookback)))
    model.add(tf.keras.layers.Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(inputs, targets, validation_split=0.2, epochs=100, batch_size=1, verbose=2)
    return model


model = create_LSTM_model(inputs_train, targets_train, lookback)
validation_predictions = model.predict(inputs_validate)

targets_train = scaler.inverse_transform([targets_train])
targets_validate = scaler.inverse_transform([targets_validate])
validation_predictions = scaler.inverse_transform(validation_predictions)

# https://medium.com/p/c9ef0d3ef2df