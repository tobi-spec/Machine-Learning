import pandas as pd

from TimeSeriesPrediction.AnomalieDetection.LogFiles.log_anomalie_detection_keras_utils import LogParser, \
    transform_messages_to_numbers, DataBuilder, create_train_data
import numpy as np
from yaml_parser import get_hyperparameters
import matplotlib.pyplot as plt


def workflow(model):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', None)

    logs_1 = LogParser("../../data/correct/log1.txt")
    translation_table = transform_messages_to_numbers(logs_1.get_messages())

    train_inputs = create_train_data(translation_table)

    error_1 = DataBuilder("../../data/error/error_log2.txt",
                          translation_table).add_number_representation()
    test_inputs = np.array([error_1.get_numbers()])

    hyperparameters: dict = get_hyperparameters("logfile_keras_autoencoder_hyperparameter.yaml")

    model.compile(optimizer='adam', loss='mae')
    model.fit(train_inputs, train_inputs,
              epochs=hyperparameters["epochs"],
              batch_size=hyperparameters["batch_size"],
              shuffle=True)
    result = model.predict(test_inputs)

    name = hyperparameters["name"]

    differences = result[0] - test_inputs[0]
    df = pd.DataFrame({
        "difference": differences,
        "messages": error_1.get_messages().values})
    df.to_csv(f"./log_anomalie_detection_keras_{name}.csv")

    print(df.sort_values("difference", ascending=False, key=abs).head(5))

    plt.plot(test_inputs[0], "black", label="Input")
    plt.plot(result[0], "red", label="reconstruction")
    plt.legend(loc="upper left")
    plt.fill_between(np.arange(50), result[0], test_inputs[0], color='lightcoral', label="error")
    plt.savefig(f"./log_anomalie_detection_keras_{name}.png")
    plt.show()