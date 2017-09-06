import unittest
import pqkmeans
import numpy

class TestBKMeans(unittest.TestCase):
    def data_source(self, n: int):
        for i in range(n):
            for _ in range(3):
                yield [i * 100] * 5

    def test_just_constuction(self):
        bkmeans = pqkmeans.clustering.BKMeans(32, 2)

    def test_invalid_construction(self):
        self.assertRaises(Exception, lambda: pqkmeans.clustering.BKMeans(10000, 1))
        self.assertRaises(Exception, lambda: pqkmeans.clustering.BKMeans(32, 100))

    def test_fit_and_predict(self):
        bkmeans = pqkmeans.clustering.BKMeans(32, 2)
        bkmeans.fit(numpy.array(list(self.data_source(10))))
