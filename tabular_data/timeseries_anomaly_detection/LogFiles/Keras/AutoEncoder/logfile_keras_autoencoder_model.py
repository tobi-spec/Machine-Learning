from keras import Model, layers, Sequential, initializers, regularizers


class AutoEncoder(Model):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Sequential([
            layers.Dense(units=50,
                         activation="linear",
                         kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                         bias_regularizer=regularizers.L2(1e-4),
                         activity_regularizer=regularizers.L2(1e-5)
                         ),
            layers.Dense(units=25,
                         activation="linear",
                         kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                         bias_regularizer=regularizers.L2(1e-4),
                         activity_regularizer=regularizers.L2(1e-5)
                         ),
            layers.Dense(units=1,
                         activation="linear",
                         kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                         bias_regularizer=regularizers.L2(1e-4),
                         activity_regularizer=regularizers.L2(1e-5)
                         )
        ])

        self.decoder = Sequential([
            layers.Dense(units=1,
                         activation="linear",
                         kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                         bias_regularizer=regularizers.L2(1e-4),
                         activity_regularizer=regularizers.L2(1e-5)
                         ),
            layers.Dense(units=25,
                         activation="linear",
                         kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                         bias_regularizer=regularizers.L2(1e-4),
                         activity_regularizer=regularizers.L2(1e-5)
                         ),
            layers.Dense(units=50,
                         activation="linear",
                         kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                         bias_regularizer=regularizers.L2(1e-4),
                         activity_regularizer=regularizers.L2(1e-5)
                         )
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
