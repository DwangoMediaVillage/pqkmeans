import unittest
import pqkmeans
import numpy
import pipe


class TestNativeClusteringSample(unittest.TestCase):
    def setUp(self):
        self.clustering = pqkmeans.clustering.NativeClusteringSample()

    def data_source(self, n: int):
        for i in range(n):
            for _ in range(3):
                yield [i * 100] * 5

    def test_just_train_array(self):
        input_array = numpy.random.random((60, 10))
        self.clustering.fit(numpy.array(input_array))
        encoded = list(self.clustering.transform(numpy.array(input_array)))
        self.assertEqual(len(input_array), len(encoded))

    def test_fit_and_transform_generator(self):
        source = numpy.vstack((
            numpy.array(list(self.data_source(20))),
            numpy.array(list(self.data_source(20))) * 100000,
        ))

        self.clustering.fit(numpy.array(list(self.data_source(20))))

        # infinite list
        encoded = self.clustering.transform_generator(self.data_source(100000000)) | pipe.take(60) | pipe.as_list

        source = numpy.array(list(self.data_source(60)))

        # sample from each cluster
        counts = [0, 0]
        for vec, label in zip(source, encoded):
            counts[label] += 1
            if label == 0:
                self.assertLessEqual(self.clustering.distance(vec, self.clustering.min_vec),
                                     self.clustering.distance(vec, self.clustering.max_vec))
            else:
                self.assertLessEqual(self.clustering.distance(vec, self.clustering.max_vec),
                                     self.clustering.distance(vec, self.clustering.min_vec))
        self.assertEqual(counts[0], counts[1])
