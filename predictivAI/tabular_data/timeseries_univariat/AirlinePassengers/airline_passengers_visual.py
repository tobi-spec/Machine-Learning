import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("AirlinePassengers.csv", sep=";")

plt.plot(data["Passengers"])
plt.xticks(range(0, 220, 20))
plt.yticks(range(0, 1200, 100))
plt.savefig("./airlinePassengers.png")