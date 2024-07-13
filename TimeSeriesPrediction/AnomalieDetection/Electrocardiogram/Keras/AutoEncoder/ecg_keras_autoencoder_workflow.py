import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


def workflow(model):
    dataframe = pd.read_csv('../../electrocardiogram.csv', header=None)
    raw_data = dataframe.values
    labels = raw_data[:, -1]
    data = raw_data[:, 0:-1]

    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=0.2, random_state=21
    )

    model.compile(optimizer='adam', loss='mae')
    model.fit(train_data, train_data,
                    epochs=20,
                    batch_size=512,
                    shuffle=True)

    var1 = test_data[0]
    var1 = var1.reshape(1, var1.shape[0])
    result = model.predict(var1)

    plt.plot(test_data[0], "black")
    plt.plot(result[0], "red")
    plt.fill_between(np.arange(140), result[0], test_data[0], color='lightcoral')
    plt.show()
