from keras import Model, layers, optimizers, initializers
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from TimeSeriesPrediction.UnivariatTimeSeriesForecast.AirlinePassengers.airline_passengers_utilities import *

EPOCHS = 1000
LEARNING_RATE = 0.0001
BATCH_SIZE = 1
LOOK_BACK = 30
PREDICTION_START = -1
NUMBER_OF_PREDICTIONS = 80


def main():
    airline_passengers = AirlinePassengersDataSet()
    train_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = train_scaler.fit_transform(airline_passengers.get_train_data().reshape(-1, 1))

    test_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_test = test_scaler.fit_transform(airline_passengers.get_test_data().reshape(-1, 1))

    train_timeseries, train_targets = TimeSeriesGenerator(scaled_train, LOOK_BACK).create_timeseries()
    test_timeseries, test_targets = TimeSeriesGenerator(scaled_test, LOOK_BACK).create_timeseries()

    model = EncoderModel()
    model.compile(optimizer=optimizers.Adam(LEARNING_RATE), loss='mean_squared_error')
    model.fit(train_timeseries, train_targets, epochs=EPOCHS, batch_size=BATCH_SIZE)

    validation_results = validation_forecast(model, test_timeseries)

    validation = pd.DataFrame()
    validation["validation"] = test_scaler.inverse_transform([validation_results]).flatten()
    validation.index += airline_passengers.threshold + LOOK_BACK

    start_index = PREDICTION_START
    start_value = train_timeseries[start_index]
    start_value_reshaped = start_value.reshape(1, start_value.shape[0], start_value.shape[1])
    prediction_results = Forecaster(
                                    model,
                                    start_value_reshaped,
                                    NUMBER_OF_PREDICTIONS,
                                    NeuronalNetworkTypes.ATTENTION).one_step_ahead()

    prediction = pd.DataFrame()
    test = train_scaler.inverse_transform([prediction_results]).flatten()
    prediction["one_step_prediction"] = test
    prediction.index += airline_passengers.threshold + start_index

    plt.plot(airline_passengers.data["Passengers"], color="red", label="dataset")
    plt.plot(airline_passengers.get_train_data(), color="green", label="training")
    plt.plot(validation["validation"], color="blue", label="validation")
    plt.plot(prediction["one_step_prediction"], color="orange", label="one_step_prediction")
    plt.title("airline passengers prediction LSTM")
    plt.xlabel("Time[Month]")
    plt.ylabel("Passengers[x1000]")
    plt.xticks(range(0, 200, 20))
    plt.yticks(range(0, 1000, 100))
    plt.legend(loc="upper left")
    plt.savefig("./airlinePassengers_keras_encoder.png")
    plt.show()


class EncoderModel(Model):
    def __init__(self):
        super().__init__()
        self.normalize = layers.LayerNormalization(epsilon=1e-6)
        self.attention = layers.MultiHeadAttention(key_dim=256, num_heads=4, dropout=0.25)
        self.dropout = layers.Dropout(0.25)
        self.convolution1 = layers.Convolution1D(filters=4, kernel_size=1, activation="relu")
        self.convolution2 = layers.Convolution1D(filters=1, kernel_size=1)

        self.global_pooling = layers.GlobalAvgPool1D(data_format="channels_first")

        self.dense1 = layers.Dense(units=128,
                                            activation="relu",
                                            kernel_initializer=initializers.RandomNormal(stddev=0.01),
                                            bias_initializer=initializers.Zeros())
        self.dense2 = layers.Dense(units=1,
                                            activation="relu",
                                            kernel_initializer=initializers.RandomNormal(stddev=0.01),
                                            bias_initializer=initializers.Zeros())

    def call(self, inputs):
        #x1 = self.normalize(inputs)
        x1 = self.attention(inputs, inputs)
        x1 = self.dropout(x1)

        res = x1 + inputs

        #x2 = self.normalize(res)
        x2 = self.convolution1(res)
        x2 = self.dropout(x2)
        x2 = self.convolution2(x2)

        x2 = self.global_pooling(x2)

        x2 = self.dense1(x2)
        x2 = self.dense2(x2)
        return x2


if __name__ == "__main__":
    main()
