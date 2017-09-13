import unittest
import pqkmeans
import numpy

class TestPQKMeans(unittest.TestCase):
    def data_source(self, n: int):
        for i in range(n):
            yield [i * 100] * 6

    def setUp(self):
        # Train PQ encoder
        self.encoder = pqkmeans.encoder.PQEncoder(num_dim=2, Ks=10)
        self.encoder.fit(numpy.array(list(self.data_source(100))))

    def test_just_construction(self):
        pqkmeans.clustering.PQKMeans(encoder=self.encoder, k=5, iteration=10, verbose=True)


    def test_fit_and_predict(self):
        print("fit_and_predict")
        engine = pqkmeans.clustering.PQKMeans(encoder=self.encoder, k=5, iteration=10, verbose=True)
        codes = self.encoder.transform(numpy.array(list(self.data_source(100))))
        print("imakara")
        predicted = engine.fit_predict(codes)
        print("predict suruo")
        print(predicted)
