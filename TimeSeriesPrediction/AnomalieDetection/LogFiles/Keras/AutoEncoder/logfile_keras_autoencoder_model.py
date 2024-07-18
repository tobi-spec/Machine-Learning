from keras import Model, layers, Sequential


class AutoEncoder(Model):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Sequential([
            layers.Dense(50, activation="relu")])

        self.decoder = Sequential([
            layers.Dense(50, activation="relu")])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
