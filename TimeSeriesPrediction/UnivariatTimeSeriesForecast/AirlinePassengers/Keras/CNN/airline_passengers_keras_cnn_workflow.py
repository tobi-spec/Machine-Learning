import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
from keras import optimizers
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
    train_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = train_scaler.fit_transform(airline_passengers.get_train_data().reshape(-1, 1))

    test_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_test = test_scaler.fit_transform(airline_passengers.get_test_data().reshape(-1, 1))

    train_timeseries, train_targets = TimeSeriesGenerator(scaled_train, LOOK_BACK, LOOK_OUT).create_timeseries()
    test_timeseries, test_targets = TimeSeriesGenerator(scaled_test, LOOK_BACK, LOOK_OUT).create_timeseries()

    early_stopping = EarlyStopping(monitor='loss', patience=50, verbose=1, restore_best_weights=True)

    model.compile(optimizer=optimizers.Adam(LEARNING_RATE), loss='mean_squared_error')
    model.fit(train_timeseries, train_targets, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stopping])

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
                                    NeuronalNetworkTypes.ATTENTION).one_step_ahead()

    prediction = pd.DataFrame()
    test = train_scaler.inverse_transform([prediction_results]).flatten()
    prediction["one_step_prediction"] = test
    prediction.index += airline_passengers.threshold + start_index - 1

    plot_results(prediction["one_step_prediction"], validation["validation"], name)