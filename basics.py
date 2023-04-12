import numpy as np
import math
import random


class Layer:
    def __init__(self, number_of_inputs, number_of_neurons):
        self.weight = 0.1* np.random.randn(number_of_inputs, number_of_neurons)
        self.bias = np.zeros(1, number_of_neurons)

    def dense(weights:list, inputs:list, biases:int) -> int:
        self.output = np.dot(self.weights, self.inputs) + self.biases


class ActivationFuntion:
    def Linear(self, inputs: int):
        self.output = inputs

    def Step(self, inputs: int):
        if inputs > 0:
            self.output = 1
        else:
            self.output = 0
    
    def Relu(self, inputs: int):
        if inputs > 0:
            self.output = inputs
        else:
            self.output = 0 

    def Sigmoid(self, inputs: int):
        self.output = 1/(1+math.e**(-inputs))

    def Softmax(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


## Loss / Cost function
# Categorial Cross Entropy
# Sum of Squard Residuals 

## Accuracy Calculation

## Ableitungen der Funktionen?