import matplotlib.pyplot as plt
from TimeSeriesPrediction.AnomalieDetection.LogFiles.log_anomalie_detection_keras_utils import DataBuilder, LogParser, \
    transform_messages_to_numbers


def main():
    logs_1 = LogParser("data/correct/log_original.txt")
    translation_table = transform_messages_to_numbers(logs_1.get_messages())

    dataset_1 = DataBuilder("data/correct/log1.txt", translation_table).add_number_representation()
    dataset_2 = DataBuilder("data/correct/log95.txt", translation_table).add_number_representation()
    dataset_3 = DataBuilder("data/correct/log90.txt", translation_table).add_number_representation()
    dataset_4 = DataBuilder("data/correct/log100.txt", translation_table).add_number_representation()
    dataset_5 = DataBuilder("data/error/error_log2.txt", translation_table).add_number_representation()

    fig, ax = plt.subplots(3, 2)
    ax[0, 0].plot(dataset_1.get_numbers())
    ax[0, 1].plot(dataset_2.get_numbers())
    ax[1, 0].plot(dataset_3.get_numbers())
    ax[1, 1].plot(dataset_4.get_numbers())
    ax[2, 0].plot(dataset_5.get_numbers())
    plt.show()


if __name__ == "__main__":
    main()



