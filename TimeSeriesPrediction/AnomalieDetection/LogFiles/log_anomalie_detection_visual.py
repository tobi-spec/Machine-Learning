import matplotlib.pyplot as plt
from TimeSeriesPrediction.AnomalieDetection.LogFiles.log_anomalie_detection_keras_utils import DataBuilder, LogParser, \
    transform_messages_to_numbers


def main():
    logs_1 = LogParser("data/correct/log_original.txt")
    translation_table = transform_messages_to_numbers(logs_1.get_messages())

    dataset_1 = DataBuilder("data/correct/log_original.txt", translation_table).add_number_representation().get_numbers()
    dataset_2 = DataBuilder("data/correct/log30.txt", translation_table).add_number_representation().get_numbers()
    dataset_3 = DataBuilder("data/correct/log40.txt", translation_table).add_number_representation().get_numbers()
    dataset_4 = DataBuilder("data/correct/log50.txt", translation_table).add_number_representation().get_numbers()
    dataset_5 = DataBuilder("data/correct/log60.txt", translation_table).add_number_representation().get_numbers()
    dataset_6 = DataBuilder("data/correct/log120.txt", translation_table).add_number_representation().get_numbers()
    dataset_8 = DataBuilder("data/error/error_log1.txt", translation_table).add_number_representation().get_numbers()
    dataset_9 = DataBuilder("data/error/error_log2.txt", translation_table).add_number_representation().get_numbers()

    fig, ax = plt.subplots(3, 3)
    ax[0, 0].plot(dataset_1)
    ax[0, 0].set_title('original')

    ax[0, 1].plot(dataset_2)
    ax[0, 1].set_title('correct')

    ax[0, 2].plot(dataset_3)
    ax[0, 2].set_title('correct')

    ax[1, 0].plot(dataset_4)
    ax[1, 0].set_title('correct')

    ax[1, 1].plot(dataset_5)
    ax[1, 1].set_title('correct')

    ax[1, 2].plot(dataset_6)
    ax[1, 2].set_title('correct')

    ax[2, 0].plot(dataset_8)
    ax[2, 0].set_title('error')

    ax[2, 1].plot(dataset_9)
    ax[2, 1].set_title('error')

    plt.show()


if __name__ == "__main__":
    main()



