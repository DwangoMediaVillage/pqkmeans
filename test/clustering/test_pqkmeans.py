import unittest
import pqkmeans
import numpy

class TestPQKMeans(unittest.TestCase):
    def data_source(self, n: int):
        for i in range(n):
            yield [i * 100] * 6

    def test_just_constuction(self):
        print([n for n in self.data_source(10)])
        print("just construction")
        enc = pqkmeans.encoder.PQEncoder(num_dim=2)
        enc.fit(numpy.array(list(self.data_source(600))))
        pqkmeans.clustering.PQKMeans(32, enc.trained_encoder.codewords)

        #pqkmeans = pqkmeans.clustering.PQKmeans(32)
        #bkmeans = pqkmeans.clustering.BKMeans(32, 2)

    def test_invalid_construction(self):
        print("vald construction")
        #self.assertRaises(Exception, lambda: pqkmeans.clustering.BKMeans(10000, 1))
        #self.assertRaises(Exception, lambda: pqkmeans.clustering.BKMeans(32, 100))

    def test_fit_and_predict(self):
        print("fit_and_predict")
        #bkmeans = pqkmeans.clustering.BKMeans(32, 2)
        #bkmeans.fit(numpy.array(list(self.data_source(10))))
