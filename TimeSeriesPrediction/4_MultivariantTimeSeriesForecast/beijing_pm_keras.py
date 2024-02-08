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

scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(dataset)

dataset = pd.DataFrame(scaled)
dataset.columns = ['pollution', 'dew', 'temperature', 'pressure', 'wind_direction', 'wind_speed', 'snow', 'rain']
dataset["target"] = dataset.loc[:, "pollution"].shift(-1)
dataset.dropna(inplace=True)

hours_of_year = 365*24
train = dataset.loc[:hours_of_year, :]
train_X = train.loc[:, ['pollution', 'dew', 'temperature', 'pressure', 'wind_direction', 'wind_speed', 'snow', 'rain']]
train_y = train.loc[:, "target"]

test = dataset.loc[hours_of_year:, :]
test_X = test.loc[:, ['pollution', 'dew', 'temperature', 'pressure', 'wind_direction', 'wind_speed', 'snow', 'rain']]
test_y = test.loc[:, "target"]

train_X = train_X.to_numpy().reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.to_numpy().reshape((test_X.shape[0], 1, test_X.shape[1]))

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(50))
model.add(tf.keras.layers.Dense(1))
model.compile(loss='mae', optimizer='adam')
model.fit(train_X, train_y, epochs=10, batch_size=72)

predictions = model.predict(test_X)

results = pd.DataFrame()
results["target"] = test_y
results["prediction"] = predictions
results.to_csv("./beijing_results.csv")

plt.plot(results["target"].head(150))
plt.plot(results["prediction"].head(150))
plt.xlabel("time")
plt.ylabel("Pollution[pm 2.5]")
plt.legend(loc="upper left")
plt.savefig("./beijing_results.png")
plt.show()

# https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
# https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-scaler
