from keras import Model, layers, initializers
from TimeSeriesPrediction.UnivariatTimeSeriesForecast.AirlinePassengers.Keras.workflows.airline_passengers_basic_workflow import workflow


class FeedForwardModel(Model):
    def __init__(self):
        super().__init__()
        self.dense1 = layers.Dense(units=50,
                                            activation="relu",
                                            kernel_initializer=initializers.RandomNormal(stddev=0.01),
                                            bias_initializer=initializers.Zeros()
                                            )
        self.dense2 = layers.Dense(units=50,
                                            activation="relu",
                                            kernel_initializer=initializers.RandomNormal(stddev=0.01),
                                            bias_initializer=initializers.Zeros())

        self.dense3 = layers.Dense(units=1,
                                            activation="relu",
                                            kernel_initializer=initializers.RandomNormal(stddev=0.01),
                                            bias_initializer=initializers.Zeros())

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return x


if __name__ == "__main__":
    model = FeedForwardModel()
    workflow(model, "ff")