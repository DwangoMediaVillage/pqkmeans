import typing
import numpy
import sklearn

import _pqkmeans


class PQKMeans(sklearn.base.BaseEstimator, sklearn.base.ClusterMixin):
    def __init__(self, encoder, k, iteration=10, verbose=False):
        super(PQKMeans, self).__init__()
        self._impl = _pqkmeans.PQKMeans(encoder.codewords, k, iteration, verbose)

    def predict_generator(self, x_test):
        #type (typing.Iterable[typing.Iterable[numpy.uint8]]) -> Any
        for vec in x_test:
            yield self._impl.predict_one(vec)

    def fit(self, x_train):
        #type (typing.Iterable[typing.Iterable[numpy.uint8]]) -> None
        assert len(x_train.shape) == 2
        self._impl.fit(x_train.reshape(-1)) # Convert to a long 1D array

    def predict(self, x_test):
        # type: (numpy.array) -> Any
        assert len(x_test.shape) == 2
        return numpy.array(list(self.predict_generator(x_test)))

    @property
    def labels_(self):
        return self._impl.labels_

    @property
    def cluster_centers_(self):
        return self._impl.cluster_centers_

