import unittest
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
from airline_passengers_utilities import *
from keras import Model


class MyTestCase(unittest.TestCase):
    def test_AirlinePassengersDataSet_constructor(self):
        airline_passengers_data = AirlinePassengersDataSet()
        self.assertTrue(isinstance(airline_passengers_data.data, pd.DataFrame))
        self.assertEqual(107, airline_passengers_data.threshold)

    def test_AirlinePassengersDataSet_train_test_types(self):
        airline_passengers_data = AirlinePassengersDataSet()
        self.assertTrue(isinstance(airline_passengers_data.test_data, np.ndarray))

    def test_AirlinePassengersDataSet_train_test_size(self):
        airline_passengers_data = AirlinePassengersDataSet()
        test_data_size = len(airline_passengers_data.data) - airline_passengers_data.threshold
        self.assertEqual(airline_passengers_data.threshold + 1, airline_passengers_data.train_data.size)
        self.assertEqual(test_data_size - 1, airline_passengers_data.test_data.size)

    def test_TimesSeriesGenerator_constructor(self):
        data = np.arange(10, 101, 10)
        timeseries = TimeSeriesGenerator(data, 10, 10)
        self.assertTrue(isinstance(timeseries.data, np.ndarray))
        self.assertEqual(10, timeseries.lookback)
        self.assertEqual(10, timeseries.lookout)

    def test_TimesSeriesGenerator_create_timeseries_only_lookback(self):
        data = np.arange(0, 51, 1)
        inputs, targets = TimeSeriesGenerator(data, 10, 0).create_timeseries()
        self.assertEqual((40, 10), inputs.shape)
        self.assertEqual((40, 0), targets.shape)

    def test_TimeSeriesGenerator_create_timeseries(self):
        data = np.arange(0, 101, 5)
        inputs, targets = TimeSeriesGenerator(data, 10, 5).create_timeseries()
        self.assertEqual((5, 10), inputs.shape)
        self.assertEqual((5, 5), targets.shape)

    def test_TimeSeriesGenerator_get_target(self):
        data = np.arange(0, 51, 1)
        timeseries = TimeSeriesGenerator(data, 10, 10)
        targets = timeseries._get_targets(30)
        self.assertEqual((10, ), targets.shape)
        self.assertEqual(30, targets[0])

    def test_TimeSeriesGenerator_get_timeseries(self):
        data = np.arange(0, 51, 1)
        timeseries = TimeSeriesGenerator(data, 10, 10)
        inputs = timeseries._get_timeseries(30)
        self.assertEqual((10, ), inputs.shape)
        self.assertEqual(20, inputs[0])


if __name__ == '__main__':
    unittest.main()
