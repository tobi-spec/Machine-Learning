import numpy as np
from keras import layers, Sequential, optimizers, losses
import idx2numpy
import matplotlib.pyplot as plt
from yaml_parser import get_hyperparameters


class MNISTDataLoader:
    def __init__(self):
        self.train_images: np.ndaray = idx2numpy.convert_from_file("../../data/train-images.idx3-ubyte").astype(
            np.float32)
        self.train_labels: np.ndaray = idx2numpy.convert_from_file("../../data/train-labels.idx1-ubyte")
        self.test_images: np.ndaray = idx2numpy.convert_from_file("../../data/t10k-images.idx3-ubyte").astype(
            np.float32)
        self.test_labels: np.ndaray = idx2numpy.convert_from_file("../../data/t10k-labels.idx1-ubyte")

    def display_train_image(self, position: int) -> None:
        plt.imshow(self.train_images[position], cmap=plt.cm.binary)
        plt.show()

    def display_test_image(self, position: int) -> None:
        plt.imshow(self.test_images[position], cmap=plt.cm.binary)
        plt.show()


mnist = MNISTDataLoader()
hyperparameters: dict = get_hyperparameters("./digit_classification_keras_ff_hyperparameter.yaml")

model = Sequential()
model.add(layers.Flatten(input_shape=(28, 28)))
model.add(layers.Dense(units=128, activation="relu"))
model.add(layers.Dense(units=10, activation="softmax"))

model.compile(optimizer=optimizers.Adam(hyperparameters["learning_rate"]), loss=losses.SparseCategoricalCrossentropy())
model.fit(mnist.train_images, mnist.train_labels, epochs=hyperparameters["epochs"],
          batch_size=hyperparameters["batch_size"])

random_images = np.random.randint(low=0, high=10000, size=10)

for image_index in random_images:
    prediction: list = model.predict(mnist.test_images[[image_index]])  # add dimension for correct input shape
    print("correct number ", mnist.test_labels[image_index])
    print("predicted number: ", np.argmax(prediction))
