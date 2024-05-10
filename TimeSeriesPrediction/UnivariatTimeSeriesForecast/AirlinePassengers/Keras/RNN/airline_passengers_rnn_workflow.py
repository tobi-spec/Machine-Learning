from keras import optimizers
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from TimeSeriesPrediction.UnivariatTimeSeriesForecast.AirlinePassengers.airline_passengers_utilities import *

EPOCHS = 600
LEARNING_RATE = 0.001
BATCH_SIZE = 1
LOOK_BACK = 30
PREDICTION_START = -1
NUMBER_OF_PREDICTIONS = 80


def workflow(model):
    airline_passengers = AirlinePassengersDataSet()
    train_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = train_scaler.fit_transform(airline_passengers.get_train_data().reshape(-1, 1))

    test_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_test = test_scaler.fit_transform(airline_passengers.get_test_data().reshape(-1, 1))

    train_timeseries, train_targets = TimeSeriesGenerator(scaled_train, LOOK_BACK).create_timeseries()
    test_timeseries, test_targets = TimeSeriesGenerator(scaled_test, LOOK_BACK).create_timeseries()

    train_timeseries = train_timeseries.reshape(train_timeseries.shape[0],train_timeseries.shape[2], train_timeseries.shape[1])
    test_timeseries = test_timeseries.reshape(test_timeseries.shape[0], test_timeseries.shape[2], test_timeseries.shape[1])

    model.compile(optimizer=optimizers.Adam(LEARNING_RATE), loss='mean_squared_error')
    model.fit(train_timeseries, train_targets, epochs=EPOCHS, batch_size=BATCH_SIZE)

    validation_results = validation_forecast(model, test_timeseries)

    validation = pd.DataFrame()
    validation["validation"] = test_scaler.inverse_transform([validation_results]).flatten()
    validation.index += airline_passengers.threshold + LOOK_BACK

    start_index = PREDICTION_START
    start_value = train_timeseries[start_index]
    start_value_reshaped = start_value.reshape(1, start_value.shape[0], start_value.shape[1])
    prediction_results = Forecaster(
                                    model,
                                    start_value_reshaped,
                                    NUMBER_OF_PREDICTIONS,
                                    NeuronalNetworkTypes.LSTM).one_step_ahead()

    prediction = pd.DataFrame()
    prediction["one_step_prediction"] = train_scaler.inverse_transform([prediction_results]).flatten()
    prediction.index += airline_passengers.threshold + start_index

    plt.plot(airline_passengers.data["Passengers"], color="red", label="dataset")
    plt.plot(airline_passengers.get_train_data(), color="green", label="training")
    plt.plot(validation["validation"], color="blue", label="validation")
    plt.plot(prediction["one_step_prediction"], color="orange", label="one_step_prediction")
    plt.title("airline passengers prediction RNN")
    plt.xlabel("Time[Month]")
    plt.ylabel("Passengers[x1000]")
    plt.xticks(range(0, 200, 20))
    plt.yticks(range(0, 1000, 100))
    plt.legend(loc="upper left")
    plt.savefig("./airlinePassengers_keras_rnn.png")
    plt.show()