import numpy as np


def single_input_single_neuron(weight:int, input:int, bias: int) -> int:
    return weight*input + bias

def multiple_inputs_single_neuron(weights:list, inputs:list, bias:int) -> int:
    return np.dot(weights, inputs) + bias

# Full Layer
def multiple_inputs_mutiple_neurons(weights:list, inputs:list, biases:int) -> int:
    return np.dot(weights, inputs) + biases

## max

#activation functions
## Linear
## relu
## Sigmoid
## Softmax

# Loss function
# Categorial Cross Entropy

## Accuracy Calculation