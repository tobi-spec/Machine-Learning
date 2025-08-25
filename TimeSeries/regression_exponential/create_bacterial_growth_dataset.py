import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


initial_population = 10
growth_rate = 0.01
noise_factor = 0.05
time_steps = np.arange(500)

# Generate exponential growth data with noise
population = initial_population * np.exp(growth_rate * time_steps)
noise = np.random.normal(0, noise_factor, size=population.shape) * population
population_noisy = population + noise

data = pd.DataFrame({
    'Time': time_steps,
    'Population': population_noisy
})

data.to_csv("./bacteria_growth_data.csv")

plt.figure(figsize=(10, 6))
plt.plot(time_steps, population_noisy, label="Bacterial Growth (with noise)", marker='o')
plt.plot(time_steps, population, label="Ideal Exponential Growth", linestyle='--', color='red')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Bacterial Colony Growth Over Time')
plt.legend()
plt.grid(True)
plt.savefig("./bacteria_growth_data.png")
plt.show()
