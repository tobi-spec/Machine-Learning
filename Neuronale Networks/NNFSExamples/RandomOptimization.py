import matplotlib.pyplot as plt
import numpy as np
from nnfs.datasets import vertical_data
from NNFS import *

# Data
X, y = vertical_data(samples=100, classes=3)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap="brg")
#plt.show()

# Model
layer_1 = main.LayerDense(2, 3)
activation1 = main.ActivationReLu()
layer_2 = main.LayerDense(3, 3)
activation2 = main.ActivationSoftMax()

loss_function = main.LossCategoricalCrossEntropy()

# Helper variables
lowest_loss = 9999999 # init values
best_layer1_weights = layer_1.weights.copy()
best_layer1_bias = layer_1.bias.copy()
best_layer2_weights = layer_2.weights.copy()
best_layer2_bias = layer_2.bias.copy()

for iteration in range(1000000):

    # New set of weights/biases for iteration
    layer_1.weights = 0.05*np.random.randn(2, 3)
    layer_1.biases = 0.05*np.random.randn(1, 3)
    layer_2.weights = 0.05*np.random.randn(3, 3)
    layer_2.biases = 0.05*np.random.randn(1, 3)

    # Model forward propagation
    layer_1.forward(X)
    activation1.forward(layer_1.output)
    layer_2.forward(activation1.output)
    activation2.forward(layer_2.output)
    loss = loss_function.calculate(activation2.output, y)

    # Accuracy
    prediction = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(prediction == y)

    if loss < lowest_loss:
        print("New set of weights found, iteration:", iteration, "loss: ", loss, "accuracy", accuracy)
        best_layer1_weights = layer_1.weights.copy()
        best_layer1_bias = layer_1.bias.copy()
        best_layer2_weights = layer_2.weights.copy()
        best_layer2_bias = layer_2.bias.copy()
        lowest_loss = loss

