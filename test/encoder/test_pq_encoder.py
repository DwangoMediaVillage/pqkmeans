import unittest
import pqkmeans
import numpy
import pipe

class TestPQEncoder(unittest.TestCase):
    def data_source(self, n: int):
        for i in range(n):
            for _ in range(3):
                yield [i * 100] * 6

    def setUp(self):
        self.encoder = pqkmeans.encoder.PQEncoder(num_dim=2)

    def test_just_train_array(self):
        input_array = numpy.random.random((300, 10))
        self.encoder.fit(numpy.array(input_array))
        encoded = list(self.encoder.transform(numpy.array(input_array)))
        self.assertEqual(len(input_array), len(encoded))

    def test_fit_and_transform_generator(self):
        self.encoder.fit(numpy.array(list(self.data_source(300))))

        # infinite list
        encoded = self.encoder.transform_generator(self.data_source(100000000)) | pipe.take(60) | pipe.as_list

        for i in range(0, len(encoded), 3):
            numpy.testing.assert_array_almost_equal(encoded[i], encoded[i+1])
            numpy.testing.assert_array_almost_equal(encoded[i], encoded[i+2])

    def test_transform_and_inverse_transform(self):
        input_array = numpy.random.random((300, 10))
        self.encoder.fit(numpy.array(input_array))
        encoded = self.encoder.transform(numpy.array(input_array))
        decoded = self.encoder.inverse_transform(encoded)

        N1, M = encoded.shape
        N2, D = decoded.shape
        self.assertEqual(N1, N2)
        self.assertEqual(M, self.encoder.M)
        self.assertEqual(D, self.encoder.Ds * self.encoder.M)
        self.assertEqual(encoded.dtype, self.encoder.code_dtype)
