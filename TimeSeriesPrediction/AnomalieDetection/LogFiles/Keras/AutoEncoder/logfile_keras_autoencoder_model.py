from keras import Model, layers, Sequential, initializers


class AutoEncoder(Model):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Sequential([
            layers.Dense(50, activation="linear",
                         kernel_initializer=initializers.GlorotNormal(),
                         bias_initializer="zeros"),
            layers.Dense(50, activation="linear",
                         kernel_initializer=initializers.GlorotNormal(),
                         bias_initializer="zeros")])

        self.decoder = Sequential([
            layers.Dense(50, activation="linear",
                         kernel_initializer=initializers.GlorotNormal(),
                         bias_initializer="zeros"),
            layers.Dense(50, activation="linear",
                         kernel_initializer=initializers.GlorotNormal(),
                         bias_initializer="zeros")])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
