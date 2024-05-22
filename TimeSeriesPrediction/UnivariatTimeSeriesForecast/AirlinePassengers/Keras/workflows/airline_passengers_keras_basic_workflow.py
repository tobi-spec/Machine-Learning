from keras import optimizers
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from TimeSeriesPrediction.UnivariatTimeSeriesForecast.AirlinePassengers.airline_passengers_utilities import *

EPOCHS = 1000
LEARNING_RATE = 0.0001
BATCH_SIZE = 1
LOOK_BACK = 30
LOOK_OUT = 1
PREDICTION_START = -1
NUMBER_OF_PREDICTIONS = 60


def workflow(model, name):
    airline_passengers = AirlinePassengersDataSet()
    train = airline_passengers.get_train_data()
    test = airline_passengers.get_test_data()

    train_inputs, train_targets = TimeSeriesGenerator(train, LOOK_BACK, LOOK_OUT).create_timeseries()
    test_inputs, test_targets = TimeSeriesGenerator(test, LOOK_BACK, LOOK_OUT).create_timeseries()

    early_stopping = EarlyStopping(monitor='loss', patience=50, verbose=1, restore_best_weights=True)

    model.compile(optimizer=optimizers.Adam(LEARNING_RATE), loss='mean_squared_error')
    model.fit(train_inputs, train_targets, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stopping])

    validation_results = validation_forecast(model, test_inputs)

    validation = pd.DataFrame()
    validation["validation"] = validation_results
    validation.index += airline_passengers.threshold + LOOK_BACK

    start_index = PREDICTION_START
    start_value = train_inputs[start_index]
    start_value_reshaped = start_value.reshape(1, start_value.shape[0])
    prediction_results = Forecaster(model,
                                    start_value_reshaped,
                                    NUMBER_OF_PREDICTIONS,
                                    NeuronalNetworkTypes.FEED_FORWARD).one_step_ahead()

    prediction = pd.DataFrame()
    prediction["one_step_prediction"] = prediction_results
    prediction.index += airline_passengers.threshold + start_index - 1

    plt.plot(airline_passengers.data["Passengers"], color="red", label="dataset")
    plt.plot(airline_passengers.get_train_data(), color="green", label="training")
    plt.plot(validation["validation"], color="blue", label="validation")
    plt.plot(prediction["one_step_prediction"], color="orange", label="one_step_prediction")
    plt.title(f"airline passengers {name}")
    plt.xlabel("Time[Month]")
    plt.ylabel("Passengers[x1000]")
    plt.xticks(range(0, 200, 20))
    plt.yticks(range(0, 1000, 100))
    plt.legend(loc="upper left")
    plt.savefig(f"../results/airlinePassengers_keras_{name}.png")
    plt.show()