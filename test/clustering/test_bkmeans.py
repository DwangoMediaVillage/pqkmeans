import unittest
import pqkmeans
import numpy
import collections


class TestBKMeans(unittest.TestCase):
    def data_source(self, n: int, bits):
        for i in range(n):
            for _ in range(3):
                datum = numpy.zeros((bits,), dtype=int)
                datum[:i] = 1
                yield datum

    def test_just_constuction(self):
        bkmeans = pqkmeans.clustering.BKMeans(k=2, input_dim=32, subspace_dim=2)

    def test_invalid_construction(self):
        self.assertRaises(Exception, lambda: pqkmeans.clustering.BKMeans(10000, 1))
        self.assertRaises(Exception, lambda: pqkmeans.clustering.BKMeans(32, 100))

    def test_fit_and_predict(self):
        bkmeans = pqkmeans.clustering.BKMeans(k=2, input_dim=32, subspace_dim=2)
        data = numpy.array(list(self.data_source(10, bits=32)))
        predicted = bkmeans.fit_predict(data)

        count = collections.defaultdict(int)
        for cluster in predicted:
            count[cluster] += 1

        self.assertGreaterEqual(min(count.values()), max(count.values()) * 0.95)

        a = bkmeans.predict(numpy.ones((1,32), dtype=int))
        b = bkmeans.predict(numpy.ones((1,32), dtype=int))
        self.assertEqual(a, b)

        self.assertRaises(Exception, lambda: bkmeans.predict(numpy.ones((1,33), dtype=int)))
