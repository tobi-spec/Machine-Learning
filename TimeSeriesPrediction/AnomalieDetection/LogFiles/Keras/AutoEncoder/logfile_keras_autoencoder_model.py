from keras import Model, layers, Sequential


class AutoEncoder(Model):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Sequential([
            layers.Dense(50, activation="linear"),
            layers.Dense(50, activation="linear")])

        self.decoder = Sequential([
            layers.Dense(50, activation="linear"),
            layers.Dense(50, activation="linear")])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
