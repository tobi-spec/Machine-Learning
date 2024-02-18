import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import matplotlib.pyplot as plt


class BeijingDataSet:
    def __init__(self):
        self.dataset = pd.read_csv(
            filepath_or_buffer="BeijingParticulateMatter.csv",
            delimiter=",",
            index_col=0,
            parse_dates=[[1, 2, 3, 4]],
            date_format='%Y %m %d %H')

        self.dataset.drop(["No"], axis=1, inplace=True)
        self.dataset.columns = ['pollution', 'dew', 'temperature', 'pressure', 'wind_direction', 'wind_speed', 'snow', 'rain']
        self.dataset.index.name = 'date'
        self.dataset.loc[:, "pollution"].fillna(0, inplace=True)
        self.dataset = self.dataset[24:]

    def encode_labels(self):
        encoder = LabelEncoder()
        self.dataset.loc[:, "wind_direction"] = encoder.fit_transform(self.dataset.loc[:, "wind_direction"])

    def add_targets(self):
        self.dataset["target"] = self.dataset.loc[:, "pollution"].shift(-1)
        self.dataset.dropna(inplace=True)

    def save(self):
        self.dataset.to_csv("./beijing_pollution.csv")

    def return_dataframe(self):
        return self.dataset


dataset = BeijingDataSet()
dataset.save()
dataset.encode_labels()
dataset.add_targets()
dataset = dataset.return_dataframe()

class Normalizer:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def transform(self, dataset):
        columns = dataset.columns
        scaled = self.scaler.fit_transform(dataset)
        dataset = pd.DataFrame(scaled)
        dataset.columns = columns
        return dataset

    def retransform(self, dataset):
        columns = dataset.columns
        dataset = self.scaler.inverse_transform(dataset)
        dataset = pd.DataFrame(dataset)
        dataset.columns = columns
        return dataset


normalizer = Normalizer()
dataset = normalizer.transform(dataset)


def create_inputs_targets(data):
    inputs = data.loc[:,
              ['pollution', 'dew', 'temperature', 'pressure', 'wind_direction', 'wind_speed', 'snow', 'rain']]
    inputs.reset_index(inplace=True, drop=True)
    targets = data.loc[:, "target"]
    return inputs, targets


hours_of_year = 365 * 24
train = dataset.loc[:hours_of_year, :]
train_X, train_y = create_inputs_targets(train)

test = dataset.loc[hours_of_year:, :]
test_X, test_y = create_inputs_targets(test)

def create_timeseries(inputs, targets, span):
    timeseries = list()
    for i in range(span, len(inputs)):
        timeseries.append(inputs.loc[i - span:i, :])
    timeseries = np.array(timeseries)
    targets = targets[span:]
    return timeseries, targets


size_of_timespan = 10
train_X_timeseries, train_y = create_timeseries(train_X, train_y, size_of_timespan)
test_X_timeseries, test_y = create_timeseries(test_X, test_y, size_of_timespan)

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(50))
model.add(tf.keras.layers.Dense(1))
model.compile(loss='mae', optimizer='adam')
model.fit(train_X_timeseries, train_y, epochs=10, batch_size=16)

predictions = model.predict(test_X_timeseries)
predictions = pd.DataFrame(predictions, columns=["predictions"])

results = test[size_of_timespan-1:]
results.reset_index(inplace=True, drop=True)
results.loc[:, "pollution"] = predictions.loc[:, "predictions"]
results.rename(columns={"pollution": "predictions"}, inplace=True)
results = normalizer.retransform(results)
results = results.loc[:, ["predictions", "target"]]
results.to_csv("./beijing_results.csv")

plt.plot(results["target"].head(150), label="test_data", color="blue")
plt.plot(results["predictions"].head(150), label="prediction_data", color="orange")
plt.xlabel("time")
plt.ylabel("Pollution[pm 2.5]")
plt.legend(loc="upper right")
plt.savefig("./beijing_results.png")
plt.show()

# https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
# https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-scaler
