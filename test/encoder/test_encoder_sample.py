# -*- coding:utf-8 -*-
import unittest
import pqkmeans
import numpy


class TestEncoderSample(unittest.TestCase):
    def data_source(self, n: int):
        for i in range(n):
            for _ in range(3):
                yield [i * 100]

    def setUp(self):
        self.encoder = pqkmeans.encoder.EncoderSample()

    def test_fit_and_predict_generator(self):
        self.encoder.fit_generator(self.data_source(20))

        # can handle infinite list
        inf = 1000000000
        for i, original, encoded, decoded in zip(
                range(inf),
                self.data_source(inf),
                self.encoder.transform_generator(self.data_source(inf)),
                self.encoder.inverse_transform_generator(self.encoder.transform_generator(self.data_source(inf)))
        ):
            self.assertEqual(original[0], decoded[0])
            if i == 10:
                break

    def test_fit_and_predict_array(self):
        input_array = numpy.random.random((3,10))
        self.encoder.fit(input_array)
        encoded = self.encoder.transform(input_array)
        self.assertEqual(input_array.shape[0], encoded.shape[0])
        decoded = self.encoder.inverse_transform(encoded)
        numpy.testing.assert_array_almost_equal(input_array, decoded)
