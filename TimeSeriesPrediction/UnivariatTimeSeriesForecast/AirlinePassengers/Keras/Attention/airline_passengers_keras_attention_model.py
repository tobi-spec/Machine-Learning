from keras import Model, layers, initializers



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
        x1 = self.attention(inputs, inputs)

        res = x1 + inputs

        x2 = self.convolution1(res)
        x2 = self.convolution2(x2)

        x2 = self.global_pooling(x2)

        x2 = self.dense1(x2)
        x2 = self.dense2(x2)
        return x2

