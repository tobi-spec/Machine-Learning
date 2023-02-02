import unittest
import basics
import numpy

class BasicsTests(unittest.TestCase):

    def test_single_input_single_neuron(self):
        actual = basics.single_input_single_neuron(weight=0.2, input=1, bias=3)
        expected = 3.2
        self.assertEqual(actual, expected)

    def test_multiple_inputs_single_neuron(self):
        weights = [0.2, 0.8, -0.5, 1.0]
        inputs = [1, 2, 3, 2.5]
        actual = basics.multiple_inputs_single_neuron(weights, inputs,  bias=3)
        expected = 5.8
        self.assertEqual(actual, expected)

    def test_multiple_inputs_multiple_neuron(self):
        weights = [
            [0.2, 0.8, -0.5, 1.0],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]
        ]
        inputs = [1, 2, 3, 2.5]
        biases = [2, 3, 0.5]

        actual = basics.multiple_inputs_mutiple_neurons(weights, inputs,  biases)
        expected = [4.8, 1.21, 2.385]

        self.assertTrue(numpy.alltrue(actual == expected))

if __name__ == '__main__':
    unittest.main()