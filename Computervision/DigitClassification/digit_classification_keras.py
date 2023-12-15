import numpy as np
import tensorflow as tf
import idx2numpy
import matplotlib.pyplot as plt
import timeit

start = timeit.default_timer()

class MNISTDataLoader:
    def __init__(self):
        self.train_images = idx2numpy.convert_from_file("./MNIST/train-images.idx3-ubyte")
        self.train_labels = tf.keras.utils.to_categorical(idx2numpy.convert_from_file("./MNIST/train-labels.idx1-ubyte"))
        self.test_images = idx2numpy.convert_from_file("./MNIST/t10k-images.idx3-ubyte")
        self.test_labels = tf.keras.utils.to_categorical(idx2numpy.convert_from_file("./MNIST/t10k-labels.idx1-ubyte"))

    def info(self):
        print("Number of images for training: ", len(self.train_images), self.train_images.shape)
        print("Number of labels for training: ", len(self.train_labels), self.train_labels.shape)
        print("Number of images for testing: ", len(self.test_images), self.test_images.shape)
        print("Number of labels for testing: ", len(self.test_labels), self.test_labels.shape)

    def display_train_image(self, position):
        plt.imshow(self.train_images[position], cmap=plt.cm.binary)
        plt.show()

    def display_test_image(self, position):
        plt.imshow(self.test_images[position], cmap=plt.cm.binary)
        plt.show()


mnist = MNISTDataLoader()
mnist.info()

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(units=128, activation="relu", kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01), bias_initializer=tf.keras.initializers.Zeros()))
model.add(tf.keras.layers.Dense(units=10, activation="softmax", kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01), bias_initializer=tf.keras.initializers.Zeros()))
model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=tf.keras.losses.CategoricalCrossentropy())
model.fit(mnist.train_images, mnist.train_labels, epochs = 15, batch_size=32)

prediction = model.predict(mnist.test_images[6758:6759])
stop = timeit.default_timer()
print("predicted number: ", np.argmax(prediction))
print("correct number ",mnist.test_labels[6758:6759])
print(f"run time[s]: {stop-start}")
