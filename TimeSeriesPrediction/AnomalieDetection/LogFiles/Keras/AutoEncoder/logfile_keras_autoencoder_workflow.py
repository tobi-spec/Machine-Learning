import pandas as pd

from TimeSeriesPrediction.AnomalieDetection.LogFiles.log_anomalie_detection_keras_utils import LogParser, \
    transform_messages_to_numbers, DataBuilder
import numpy as np
from yaml_parser import get_hyperparameters
import matplotlib.pyplot as plt


def workflow(model):
    logs_1 = LogParser("../../data/log_example.txt")
    translation_table = transform_messages_to_numbers(logs_1.get_messages())

    timeseries_1 = DataBuilder("../../data/log_example.txt",
                               translation_table).add_number_representation().get_numbers()
    timeseries_2 = DataBuilder("../../data/log_example2.txt",
                               translation_table).add_number_representation().get_numbers()
    timeseries_3 = DataBuilder("../../data/log_example3.txt",
                               translation_table).add_number_representation().get_numbers()
    timeseries_4 = DataBuilder("../../data/log_example4.txt",
                               translation_table).add_number_representation().get_numbers()

    train_inputs = np.array([timeseries_1, timeseries_2, timeseries_3, timeseries_4])

    error_1 = DataBuilder("../../data/log_error1.txt", translation_table).add_number_representation().get_numbers()
    test_inputs = np.array([error_1])

    hyperparameters: dict = get_hyperparameters("logfile_keras_autoencoder_hyperparameter.yaml")

    model.compile(optimizer='adam', loss='mae')
    model.fit(train_inputs, train_inputs,
              epochs=hyperparameters["epochs"],
              batch_size=hyperparameters["batch_size"],
              shuffle=True)
    result = model.predict(test_inputs)

    plt.plot(error_1, "black")
    plt.plot(result[0], "red")
    #plt.fill_between(np.arange(50), result, test_inputs, color='lightcoral')
    plt.show()
