from keras import optimizers
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from TimeSeries.UnivariatTimeSeriesForecast.AirlinePassengers.airline_passengers_utilities import *
from yaml_parser import get_hyperparameters


def workflow(model):
    airline_passengers = AirlinePassengersDataSet()
    train_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = train_scaler.fit_transform(airline_passengers.train_data.reshape(-1, 1))

    test_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_test = test_scaler.fit_transform(airline_passengers.test_data.reshape(-1, 1))

    hyperparameters: dict = get_hyperparameters("airline_passengers_keras_lstm_hyperparameter.yaml")

    train_timeseries, train_targets = TimeSeriesGenerator(scaled_train,  hyperparameters["look_back"], hyperparameters["look_out"]).create_timeseries()
    test_timeseries, test_targets = TimeSeriesGenerator(scaled_test,  hyperparameters["look_back"], hyperparameters["look_out"]).create_timeseries()

    train_timeseries = train_timeseries.reshape(train_timeseries.shape[0],train_timeseries.shape[2], train_timeseries.shape[1])
    test_timeseries = test_timeseries.reshape(test_timeseries.shape[0], test_timeseries.shape[2], test_timeseries.shape[1])

    early_stopping = EarlyStopping(monitor='loss', patience=75, verbose=1, restore_best_weights=True)

    model.compile(optimizer=optimizers.Adam(hyperparameters["learning_rate"]), loss='mean_squared_error')
    model.fit(train_timeseries, train_targets, epochs=hyperparameters["epochs"], batch_size=hyperparameters["batch_size"], callbacks=[early_stopping])

    validation_results = keras_forecast(model, test_timeseries)

    validation = pd.DataFrame()
    validation["validation"] = test_scaler.inverse_transform([validation_results]).flatten()
    validation.index += airline_passengers.threshold + hyperparameters["look_back"]

    start_index = -1
    start_value = train_timeseries[start_index]
    start_value_reshaped = start_value.reshape(1, start_value.shape[0], start_value.shape[1])
    prediction_results = KerasForecaster(
                                    model,
                                    start_value_reshaped,
                                    hyperparameters["number_of_predictions"],
                                    NeuronalNetworkTypes.LSTM).one_step_ahead()

    prediction = pd.DataFrame()
    prediction["one_step_prediction"] = train_scaler.inverse_transform([prediction_results]).flatten()
    prediction.index += airline_passengers.threshold + start_index - 1

    plot_results(prediction["one_step_prediction"], validation["validation"], hyperparameters["name"])