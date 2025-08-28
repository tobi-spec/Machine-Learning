import idx2numpy
import numpy as np
from matplotlib import pyplot as plt


class MNISTDataLoader:
    def __init__(self):
        self.train_images: np.ndaray = idx2numpy.convert_from_file("data/train-images.idx3-ubyte").astype(
            np.float32)
        self.train_labels: np.ndaray = idx2numpy.convert_from_file("data/train-labels.idx1-ubyte")
        self.test_images: np.ndaray = idx2numpy.convert_from_file("data/t10k-images.idx3-ubyte").astype(
            np.float32)
        self.test_labels: np.ndaray = idx2numpy.convert_from_file("data/t10k-labels.idx1-ubyte")

    def display_train_image(self, position: int) -> None:
        plt.imshow(self.train_images[position], cmap=plt.cm.binary)
        plt.show()

    def display_test_image(self, position: int) -> None:
        plt.imshow(self.test_images[position], cmap=plt.cm.binary)
        plt.show()


MNISTDataLoader().display_test_image(100)
