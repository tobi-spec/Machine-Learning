import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import matplotlib.pyplot as plt


dataset = pd.read_csv(
    filepath_or_buffer="BeijingParticulateMatter.csv",
    delimiter=",",
    index_col=0,
    parse_dates=[[1, 2, 3, 4]],
    date_format='%Y %m %d %H')

dataset.drop(["No"], axis=1, inplace=True)
dataset.columns = ['pollution', 'dew', 'temperature', 'pressure', 'wind_direction', 'wind_speed', 'snow', 'rain']
dataset.index.name = 'date'
dataset["pollution"] = dataset.loc[:, "pollution"].fillna(0)
dataset = dataset[24:]
dataset.to_csv("./beijing_pollution.csv")

encoder = LabelEncoder()
dataset.loc[:, "wind_direction"] = encoder.fit_transform(dataset.loc[:, "wind_direction"])

dataset["target"] = dataset.loc[:, "pollution"].shift(-1)
dataset.dropna(inplace=True)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(dataset)
dataset = pd.DataFrame(scaled)
dataset.columns = ['pollution', 'dew', 'temperature', 'pressure', 'wind_direction', 'wind_speed', 'snow', 'rain',
                   "target"]

hours_of_year = 365 * 24
train = dataset.loc[:hours_of_year, :]
train_X = train.loc[:, ['pollution', 'dew', 'temperature', 'pressure', 'wind_direction', 'wind_speed', 'snow', 'rain']]
train_y = train.loc[:, "target"]

test = dataset.loc[hours_of_year:, :]
test_X = test.loc[:, ['pollution', 'dew', 'temperature', 'pressure', 'wind_direction', 'wind_speed', 'snow', 'rain']]
test_X.reset_index(inplace=True, drop=True)
test_y = test.loc[:, "target"]

train_X_timeseries = list()
for i in range(1, len(train_X)):
    train_X_timeseries.append(train_X.loc[i - 1:i, :])
train_X_timeseries = np.array(train_X_timeseries)

test_X_timeseries = list()
for i in range(1, len(test_X)):
    test_X_timeseries.append(test_X.loc[i - 1:i, :])
test_X_timeseries = np.array(test_X_timeseries)

train_y = train_y[1:]
test_y = test_y[1:]

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(50))
model.add(tf.keras.layers.Dense(1))
model.compile(loss='mae', optimizer='adam')
model.fit(train_X_timeseries, train_y, epochs=10, batch_size=16)

predictions = model.predict(test_X_timeseries)

predictions = pd.DataFrame(predictions, columns=["pollution_prediction"])
test.reset_index(inplace=True, drop=True)
predictions.reset_index(inplace=True, drop=True)
test.loc[:, "pollution"] = predictions.loc[:, "pollution_prediction"]
test = scaler.inverse_transform(test)
test = pd.DataFrame(test)
test.columns = ['pollution_prediction', 'dew', 'temperature', 'pressure', 'wind_direction', 'wind_speed', 'snow',
                'rain', "target"]

results = pd.DataFrame()
results["target"] = test["target"]
results["prediction"] = test["pollution_prediction"]
results.to_csv("./beijing_results.csv")

plt.plot(results["target"].head(150), label="test_data", color="blue")
plt.plot(results["prediction"].head(150), label="prediction_data", color="orange")
plt.xlabel("time")
plt.ylabel("Pollution[pm 2.5]")
plt.legend(loc="upper right")
plt.savefig("./beijing_results.png")
plt.show()

# https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
# https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-scaler
