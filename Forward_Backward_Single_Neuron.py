## forward and backward pass of single neuron
# Decreasing of Relu activation function as example

inputs = [1.0, -2.0, 3.0]
weights = [-3.0, -1.0, 2.0]
bias = 1

#### Forward Pass
# multiply weights with inputs - calculation neuron inputs
xw0 = inputs[0] * weights[0]
xw1 = inputs[1] * weights[1]
xw2 = inputs[2] * weights[2]
print("inputs * weights", xw0, xw1, xw2)

# Adding all values with bias - calculation Neuron value
z = xw0 + xw1 + xw2 + bias
print("neuron", z)

# ReLu activation function - calculation neuron output
y = max(z, 0)
print("activation function", y)

## overall function
# y = ReLu(x0 * w0 + x1 * w1 + x1 * w2 + bias)




#### Backward pass
# The derivative from the next layer
derivation_value = 1.0

# Derivative of ReLU and the chain rule (inner and outer derivation)
derivation_relu = derivation_value*(1 if z > 0 else 0)
print("activation function derivation", derivation_relu)

# Partial derivation of the sum respectiv to the paramater aka the multiplications and the bias, using the chain rule
derivation_sum_dxw0 = 1
derivation_relu_dxw0 = derivation_relu * derivation_sum_dxw0
print("partial derivation for xw0", derivation_relu_dxw0)

derivation_sum_dxw1 = 1
derivation_relu_dxw1 = derivation_relu * derivation_sum_dxw0
print("partial derivation for xw1", derivation_relu_dxw1)

derivation_sum_dxw2 = 1
derivation_relu_dxw2 = derivation_relu * derivation_sum_dxw0
print("partial derivation for xw1", derivation_relu_dxw2)

derivation_sum_bias = 1
derivation_relu_bias = derivation_relu * derivation_sum_bias
print("partial derivation for bias", derivation_relu_bias)

# derivation of the multiplications of weights and inputs
print("derivation of the multiplications of weights and inputs")

derivation_relu_dx0 = derivation_value*(1 if z > 0 else 0) * weights[0]
print(derivation_relu_dx0)

derivation_dw0 = inputs[0]
derivation_relu_dw0 = derivation_relu_dxw0*derivation_dw0
print(derivation_relu_dw0)

derivation_dx1 = weights[1]
derivation_relu_dx1 = derivation_relu_dxw1*derivation_dx1
print(derivation_relu_dx1)

derivation_dw1 = inputs[1]
derivation_relu_dw1 = derivation_relu_dxw1*derivation_dw1
print(derivation_relu_dw1)

derivation_dx2 = weights[2]
derivation_relu_dx2 = derivation_relu_dxw2*derivation_dx2
print(derivation_relu_dx2)

derivation_dw2 = inputs[2]
derivation_relu_dw2 = derivation_relu_dxw2*derivation_dw2
print(derivation_relu_dw2)

# gradients

dx = [derivation_relu_dx0, derivation_relu_dx1, derivation_relu_dx2]
dw = [derivation_relu_dw0, derivation_relu_dw1, derivation_relu_dw2]
db = derivation_relu_bias


## Applay fraction of gradients to initial weights and bias

weights[0] += -0.001* dw[0]
weights[1] += -0.001* dw[1]
weights[2] += -0.001* dw[2]
bias += -0.001*db

print("New weights and bias", weights, bias)


## Second forward pass

xw0 = inputs[0] * weights[0]
xw1 = inputs[1] * weights[1]
xw2 = inputs[2] * weights[2]
print("inputs * weights", xw0, xw1, xw2)

# Adding all values with bias - calculation Neuron value
z = xw0 + xw1 + xw2 + bias
print("neuron", z)

# ReLu activation function - calculation neuron output
y = max(z, 0)
print("activation function", y)