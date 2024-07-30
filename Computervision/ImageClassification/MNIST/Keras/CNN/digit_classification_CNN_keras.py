import numpy as np
from keras import layers, Sequential, optimizers, losses
import idx2numpy
import matplotlib.pyplot as plt

from yaml_parser import get_hyperparameters


class MNISTDataLoader:
    def __init__(self):
        self.train_images: np.ndaray = idx2numpy.convert_from_file("../../data/train-images.idx3-ubyte").astype(np.float32)
        self.train_labels: np.ndaray = idx2numpy.convert_from_file("../../data/train-labels.idx1-ubyte")
        self.test_images: np.ndaray = idx2numpy.convert_from_file("../../data/t10k-images.idx3-ubyte").astype(np.float32)
        self.test_labels: np.ndaray = idx2numpy.convert_from_file("../../data/t10k-labels.idx1-ubyte")

    def display_train_image(self, position: int) -> None:
        plt.imshow(self.train_images[position], cmap=plt.cm.binary)
        plt.show()

    def display_test_image(self, position: int) -> None:
        plt.imshow(self.test_images[position], cmap=plt.cm.binary)
        plt.show()


mnist = MNISTDataLoader()
hyperparameters: dict = get_hyperparameters("./digit_classification_keras_cnn_hyperparameter.yaml")

model = Sequential()
model.add(layers.Reshape(target_shape=(1, 28, 28)))
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), strides=2, activation="relu", data_format="channels_first"))
model.add(layers.MaxPool2D(pool_size=(2, 2)))
model.add(layers.Flatten(input_shape=(28, 28)))
model.add(layers.Dense(units=128, activation="relu"))
model.add(layers.Dense(units=10, activation="softmax"))


model.compile(optimizer=optimizers.Adam(hyperparameters["learning_rate"]), loss=losses.SparseCategoricalCrossentropy())
model.fit(mnist.train_images, mnist.train_labels, epochs=hyperparameters["epochs"], batch_size=hyperparameters["batch_size"])

prediction: list = model.predict(mnist.test_images[[6758]])
print("predicted number: ", np.argmax(prediction))
print("correct number ", mnist.test_labels[[6758]])

