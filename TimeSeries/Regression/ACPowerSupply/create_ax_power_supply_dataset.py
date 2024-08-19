import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


x = np.arange(0, np.pi * 5, 0.1)
y = np.sin(x)

data = pd.DataFrame({
    'Time': x,
    'Value': y
})

data.to_csv("./ac_power_supply_data.csv")

plt.figure(figsize=(10, 6))
plt.plot(x, y, label="Ideal Sine Wave", linestyle='--', color='red')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Noisy Sine Wave Over Time')
plt.legend()
plt.savefig("./ac_power_supply_data.png")
plt.show()
