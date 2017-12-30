import unittest
import pqkmeans
import numpy
import collections
import pickle


class TestPQKMeans(unittest.TestCase):
    def data_source(self, n: int):
        for i in range(n):
            yield [i * 100] * 6

    def setUp(self):
        # Train PQ encoder
        self.encoder = pqkmeans.encoder.PQEncoder(num_subdim=3, Ks=20)
        self.encoder.fit(numpy.array(list(self.data_source(200))))

    def test_just_construction(self):
        pqkmeans.clustering.PQKMeans(encoder=self.encoder, k=5, iteration=10, verbose=False)

    def test_fit_and_predict(self):
        engine = pqkmeans.clustering.PQKMeans(encoder=self.encoder, k=2, iteration=10, verbose=False)
        codes = self.encoder.transform(numpy.array(list(self.data_source(100))))
        predicted = engine.fit_predict(codes)

        count = collections.defaultdict(int)
        for cluster in predicted:
            count[cluster] += 1

        # roughly balanced clusters
        self.assertGreaterEqual(min(count.values()), max(count.values()) * 0.7)

        a = engine.predict(codes[0:1, :])
        b = engine.predict(codes[0:1, :])
        self.assertEqual(a, b)

    def test_cluster_centers_are_really_nearest(self):
        engine = pqkmeans.clustering.PQKMeans(encoder=self.encoder, k=2, iteration=10, verbose=False)
        codes = self.encoder.transform(numpy.array(list(self.data_source(100))))
        fit_predicted = engine.fit_predict(codes)
        cluster_centers = numpy.array(engine.cluster_centers_, dtype=numpy.uint8)
        predicted = engine.predict(codes)

        self.assertTrue((fit_predicted == predicted).all())

        # Reconstruct the original vectors
        codes_decoded = self.encoder.inverse_transform(codes)
        cluster_centers_decoded = self.encoder.inverse_transform(cluster_centers)

        for cluster, code_decoded in zip(predicted, codes_decoded):
            other_cluster = (cluster + 1) % max(predicted)
            self.assertLessEqual(
                numpy.linalg.norm(cluster_centers_decoded[cluster] - code_decoded),
                numpy.linalg.norm(cluster_centers_decoded[other_cluster] - code_decoded)
            )

    def test_constructor_with_cluster_center(self):
        # Run pqkmeans first.
        engine = pqkmeans.clustering.PQKMeans(encoder=self.encoder, k=5, iteration=10, verbose=False)
        codes = self.encoder.transform(numpy.array(list(self.data_source(100))))
        fit_predicted = engine.fit_predict(codes)
        cluster_centers = numpy.array(engine.cluster_centers_, dtype=numpy.uint8)
        predicted = engine.predict(codes)


        # save current engine and recover from savedata
        engine_savedata = pickle.dumps(engine)
        engine_recovered = pickle.loads(engine_savedata)
        fit_predicted_from_recovered_obj = engine_recovered.predict(codes)

        numpy.testing.assert_array_equal(predicted, fit_predicted_from_recovered_obj)
