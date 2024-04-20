import tensorflow as tf
import matplotlib.pyplot as plt
from TimeSeriesPrediction.UnivariatTimeSeriesForecast.AirlinePassengers.airline_passengers_utilities import *


def main():
    airline_passengers = AirlinePassengersDataSet()
    train = airline_passengers.get_train_data()
    test = airline_passengers.get_test_data()

    lookback = 30
    train_inputs, train_targets = TimeSeriesGenerator(train, lookback).create_timeseries()
    test_inputs, test_targets = TimeSeriesGenerator(test, lookback).create_timeseries()

    model = FeedForwardModel()
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss='mean_squared_error')
    model.fit(train_inputs, train_targets, epochs=1000, batch_size=1)

    validation_results = validation_forecast(model, test_inputs)

    validation = pd.DataFrame()
    validation["validation"] = validation_results
    validation.index += airline_passengers.threshold + lookback

    start_index = -1
    start_value = train_inputs[start_index]
    start_value_reshaped = start_value.reshape(1, start_value.shape[0])
    number_of_predictions = 80
    prediction_results = one_step_ahead_forecast(model, start_value_reshaped, number_of_predictions)

    prediction = pd.DataFrame()
    prediction["one_step_prediction"] = prediction_results
    prediction.index += airline_passengers.threshold + start_index

    plt.plot(airline_passengers.data["Passengers"], color="red", label="dataset")
    plt.plot(airline_passengers.get_train_data(), color="green", label="training")
    plt.plot(validation["validation"], color="blue", label="validation")
    plt.plot(prediction["one_step_prediction"], color="orange", label="one_step_prediction")
    plt.title("airline passengers prediction FF")
    plt.xlabel("Time[Month]")
    plt.ylabel("Passengers[x1000]")
    plt.xticks(range(0, 200, 20))
    plt.yticks(range(0, 1000, 100))
    plt.legend(loc="upper left")
    plt.savefig("./airlinePassengers_keras_ff.png")
    plt.show()


class FeedForwardModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units=50,
                                            activation="relu",
                                            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                                            bias_initializer=tf.keras.initializers.Zeros()
                                            )
        self.dense2 = tf.keras.layers.Dense(units=50,
                                            activation="relu",
                                            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                                            bias_initializer=tf.keras.initializers.Zeros())

        self.dense3 = tf.keras.layers.Dense(units=1,
                                            activation="relu",
                                            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                                            bias_initializer=tf.keras.initializers.Zeros())

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return x




def one_step_ahead_forecast(model, current_value, number_of_predictions):
    one_step_ahead_forecast = list()
    for element in range(0, number_of_predictions):
        prediction = model.predict(current_value)
        one_step_ahead_forecast.append(prediction[0][0])
        current_value = np.delete(current_value, 0)
        current_value = np.append(current_value, prediction)
        current_value = current_value.reshape(1, current_value.shape[0])
    return one_step_ahead_forecast


if __name__ == "__main__":
    main()
