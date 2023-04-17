import numpy as np 

# Passed in gradient from the next layer
dvalues = np.array([[1., 1., 1.],
                    [2., 2., 2.],
                    [3., 3., 3.]])

# backpropagation layer has 3 Neurons with 4 inputs each. 
# 4 Inputs each Neuron
inputs = np.array([[1, 2, 3, 2.5],
                    [2., 5., -1., 2],
                    [-1.5, 2.7, 3.3, -0.8]])

weights = np.array([[0.2, 0.8, -0.5, 1],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]]).T

biases = np.array([[2, 3, 0.5]])


# Forward Pass
layer_output = np.dot(inputs, weights) + biases
relu_output = (np.maximum(0, layer_output))

# Backward Pass
# Relu is not correct
drelu = relu_output.copy()
drelu[layer_output <= 0] = 0 


dinputs = np.dot(drelu, weights.T)

dweights = np.dot(inputs.T, drelu)

dbiases = np.sum(drelu, axis=0, keepdims=True)

weights += -0.001 * dweights
biases += -0.001 * dbiases

print(weights)
print(biases)