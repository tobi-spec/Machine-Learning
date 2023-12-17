import numpy as np
import tensorflow as tf
import idx2numpy
import matplotlib.pyplot as plt
import timeit

start = timeit.default_timer()

class MNISTDataLoader:
    def __init__(self):
        self.train_images = idx2numpy.convert_from_file("./MNIST/train-images.idx3-ubyte").astype(np.float32)
        self.train_labels = idx2numpy.convert_from_file("./MNIST/train-labels.idx1-ubyte")
        self.test_images = idx2numpy.convert_from_file("./MNIST/t10k-images.idx3-ubyte").astype(np.float32)
        self.test_labels = idx2numpy.convert_from_file("./MNIST/t10k-labels.idx1-ubyte")

    def info(self):
        print("Number of images for training: ", len(self.train_images), self.train_images.shape)
        print("Number of labels for training: ", len(self.train_labels), self.train_labels.shape)
        print("Number of images for testing: ", len(self.test_images), self.test_images.shape)
        print("Number of labels for testing: ", len(self.test_labels), self.test_labels.shape)
        print("Labels: ", set(self.train_labels))

    def display_train_image(self, position):
        plt.imshow(self.train_images[position], cmap=plt.cm.binary)
        plt.show()

    def display_test_image(self, position):
        plt.imshow(self.test_images[position], cmap=plt.cm.binary)
        plt.show()


mnist = MNISTDataLoader()
mnist.info()

model = tf.keras.Sequential()
model.add(tf.keras.layers.Reshape(target_shape=(1, 28, 28)))
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=2, activation="relu", data_format="channels_first"))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(units=128, activation="relu"))
model.add(tf.keras.layers.Dense(units=10, activation="softmax"))
model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
model.fit(mnist.train_images, mnist.train_labels, epochs = 15, batch_size=32)

results = model.evaluate(mnist.test_images, mnist.test_labels, batch_size=32)
print("____________________________")
print("validation: ", results)
print("____________________________")

prediction = model.predict(mnist.test_images[6758:6759])
stop = timeit.default_timer()
print("predicted number: ", np.argmax(prediction))
print("correct number ",mnist.test_labels[6758:6759])
print(f"run time[s]: {stop-start}")
