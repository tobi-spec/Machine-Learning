from keras import optimizers
from keras.callbacks import EarlyStopping
from predictivAI.tabular_data.timeseries_univariat.AirlinePassengers.airline_passengers_utilities import *
from yaml_parser import get_hyperparameters


def workflow(model):
    airline_passengers = AirlinePassengersDataSet()
    train = airline_passengers.train_data
    test = airline_passengers.test_data

    hyperparameters: dict = get_hyperparameters("airline_passengers_keras_ff_hyperparameter.yaml")

    train_inputs, train_targets = TimeSeriesGenerator(train, hyperparameters["look_back"], hyperparameters["look_out"]).create_timeseries()
    test_inputs, test_targets = TimeSeriesGenerator(test, hyperparameters["look_back"], hyperparameters["look_out"]).create_timeseries()

    early_stopping = EarlyStopping(monitor='loss', patience=50, verbose=1, restore_best_weights=True)

    model.compile(optimizer=optimizers.Adam(hyperparameters["learning_rate"]), loss='mean_squared_error')
    model.fit(train_inputs, train_targets, epochs=hyperparameters["epochs"], batch_size=hyperparameters["batch_size"], callbacks=[early_stopping])

    validation_results = keras_forecast(model, test_inputs)

    validation = pd.DataFrame()
    validation["validation"] = validation_results
    validation.index += airline_passengers.threshold + hyperparameters["look_back"]

    start_index = -1
    start_value = train_inputs[start_index]
    start_value_reshaped = start_value.reshape(1, start_value.shape[0])
    prediction_results = KerasForecaster(model,
                                         start_value_reshaped,
                                         hyperparameters["number_of_predictions"],
                                         NeuronalNetworkTypes.FEED_FORWARD).one_step_ahead()

    prediction = pd.DataFrame()
    prediction["one_step_prediction"] = prediction_results
    prediction.index += airline_passengers.threshold + start_index - 1

    plot_results(prediction["one_step_prediction"], validation["validation"], hyperparameters["name"])