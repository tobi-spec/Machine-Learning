import torch
import numpy as np

tensor: torch.Tensor = torch.tensor(np.array([[[-1, 2, 3], [-4, 5, 6], [-7, 8, 9]]]))
print("-------------")
print(tensor)

print("------size-------")
print(tensor.size())

print("------shape -  alias for size() to by closer to numpy-------")
print(tensor.shape)


print("------dtype-------")
print(tensor.dtype)

print("------ndim-------")
print(tensor.ndim)

print("------accessing subparts-------")
print(tensor[0][0][0])
print(tensor[0][0][1])
print(tensor[0][0][2])
print(tensor[0][1][0])
print(tensor[0][1][1])
print(tensor[0][1][2])
print(tensor[0][2][0])
print(tensor[0][2][1])
print(tensor[0][2][2])

print("------accessing subparts via loop-------")

for i in range(3):
    for j in range(3):
        val = tensor[0, i, j]
        print(f"t[{i}][{j}] =", val)

print("------get value of tensor-------")

print(tensor[0][0][0].item())

print("------reshape-------")
tensor_reshaped = tensor.reshape(1, 1, 9)
print(tensor_reshaped)

print("------flatten - copy of tensor-------")
print(tensor.flatten())

print("------ravel - original tensor-------")
print(tensor.ravel())

print("------reshape((-1,))-------")
tensor_reshaped = tensor.reshape((-1,))
print(tensor_reshaped)

print("------clip lower/higher values-------")
print(tensor.clip(min=3, max=7))

print("------to numpy-------")
print(tensor.numpy())

print("------to list-------")
print(tensor.tolist())

print("------to absolute values-------")
print(tensor.abs())

print("------multiply-------")
print(tensor.multiply(3))

print("------math functions-------")
print(tensor.sin())
print(tensor.cos())
print(tensor.tan())
print(tensor.tanh())
print(tensor.exp())
print(tensor.log())
print(tensor.sqrt())
print(tensor.acos())
print(tensor.asin())
print(tensor.atan())



