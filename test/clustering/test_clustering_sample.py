import unittest
import pqkmeans
import numpy

class TestClusteringSample(unittest.TestCase):
    def data_source(self, n: int):
        for i in range(n):
            for _ in range(3):
                yield [i * 100] * 5

    def setUp(self):
        self.clustering = pqkmeans.clustering.ClusteringSample()

    def test_just_train_array(self):
        input_array = numpy.random.random((60, 10))
        self.clustering.fit(numpy.array(input_array))
        encoded = list(self.clustering.transform(numpy.array(input_array)))
        self.assertEqual(len(input_array), len(encoded))

    # def test_fit_and_transform_generator(self):
    #     self.encoder.fit(numpy.array(list(self.data_source(20))))
    #
    #     # infinite list
    #     encoded = self.encoder.transform_generator(self.data_source(100000000)) | pipe.take(60) | pipe.as_list
    #
    #     for i in range(0, len(encoded), 3):
    #         numpy.testing.assert_array_almost_equal(encoded[i], encoded[i+1])
    #         numpy.testing.assert_array_almost_equal(encoded[i], encoded[i+2])
    #
