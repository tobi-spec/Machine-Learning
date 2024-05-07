from keras import Model, layers
from airline_passengers_cnn_workflow import workflow


class CNNModel(Model):
    def __init__(self):
        super().__init__()
        self.cnn = layers.Conv1D(filters=64,
                                 kernel_size=3,
                                 activation='relu')
        self.pooling = layers.MaxPooling1D(pool_size=2)
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(units=50, activation="relu")
        self.dense2 = layers.Dense(units=1, activation="relu")

    def call(self, inputs):
        x = self.cnn(inputs)
        x = self.pooling(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x


if __name__ == "__main__":
    cnn_model = CNNModel()
    workflow(cnn_model)