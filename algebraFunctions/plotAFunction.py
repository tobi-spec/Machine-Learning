import matplotlib.pyplot as plt


def mathFunction(x: int) -> int:
    return x**7 + x**6 + x**5 + x**4 + x**3 + x**2 + x + 3

x_values = [x for x  in range(-5, 6)]
y_values = [mathFunction(x) for x in x_values]

print(x_values, y_values)
plt.plot(x_values, y_values)
plt.show()