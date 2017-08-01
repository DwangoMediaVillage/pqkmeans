import unittest
import pqkmeans
import numpy

class TestITQEncoder(unittest.TestCase):
    def setUp(self):
        self.encoder = pqkmeans.encoder.ITQEncoder(num_bit=5)

    def test_just_train(self):
        input_array = numpy.random.random((60, 10))
        self.encoder.fit(input_array)

    def test_fit_and_predict_array(self):
        input_array = numpy.random.random((60, 10))
        self.encoder.fit(input_array)
        encoded = self.encoder.transform(input_array)
        decoded = self.encoder.inverse_transform(encoded)
        numpy.testing.assert_array_almost_equal(input_array, decoded)
