from keras import Model, layers, initializers



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



