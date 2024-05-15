from keras import Model, layers, initializers
from TimeSeriesPrediction.UnivariatTimeSeriesForecast.AirlinePassengers.Keras.workflows.airline_passengers_rnn_workflow import workflow


class RNNModel(Model):
    def __init__(self):
        super().__init__()
        self.lstm = layers.SimpleRNN(units=50,)
        self.dense1 = layers.Dense(units=1,
                                            activation="relu",
                                            kernel_initializer=initializers.RandomNormal(stddev=0.01),
                                            bias_initializer=initializers.Zeros())

    def call(self, inputs):
        x = self.lstm(inputs)
        x = self.dense1(x)
        return x


if __name__ == "__main__":
    rnn_model = RNNModel()
    workflow(rnn_model, "rnn")
