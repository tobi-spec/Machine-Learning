import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# numpy
x = [215, 325, 185, 332, 406, 522, 412, 614, 544, 421, 445, 408],
y = [14.2, 16.4, 11.9, 15.2, 18.5, 22.1, 19.4, 25.1, 23.4, 18.1, 22.6, 17.2]
res = np.corrcoef(x, y)
print(res)

# pandas
import pandas as pd

data = {'x': [45, 37, 42, 35, 39],
        'y': [38, 31, 26, 28, 33],
        'z': [10, 15, 17, 21, 12]}

dataframe = pd.DataFrame(data, columns=['x', 'y', 'z'])
print("Dataframe is : ")
print(dataframe)

matrix = dataframe.corr()
print("Correlation matrix is : ")
print(matrix)

# seaborn
matrix = dataframe.corr()
plt.figure(figsize=(8,6))
sns.heatmap(matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()