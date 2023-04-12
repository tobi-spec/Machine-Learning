import unittest
from basics import *
import numpy

class BasicsTests(unittest.TestCase):

    def test_single_neuron(self):
        number_of_inputs = 1
        number_of_neurons = 1
        layer = Layer(number_of_inputs, number_of_neurons)
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()