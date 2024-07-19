from TimeSeriesPrediction.AnomalieDetection.LogFiles.log_anomalie_detection_keras_utils import LogParser, \
    transform_messages_to_numbers, DataBuilder, create_train_data
import numpy as np
from yaml_parser import get_hyperparameters
import matplotlib.pyplot as plt


def workflow(model):
    logs_1 = LogParser("../../data/correct/log1.txt")
    translation_table = transform_messages_to_numbers(logs_1.get_messages())

    train_inputs = create_train_data(translation_table)

    error_1 = DataBuilder("../../data/error/log_error.txt",
                          translation_table).add_number_representation().get_numbers()
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
    plt.fill_between(np.arange(50), result[0], test_inputs[0], color='lightcoral')
    plt.show()
