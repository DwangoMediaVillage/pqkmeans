import typing
import numpy
import sklearn

import _pqkmeans


class BKMeans(sklearn.base.BaseEstimator, sklearn.base.ClusterMixin):
    def __init__(self, k, input_dim, subspace_dim=8, iteration=10, verbose=False):
        super(BKMeans, self).__init__()
        self._impl = _pqkmeans.BKMeans(k, input_dim, subspace_dim, iteration, verbose)

    def predict_generator(self, x_test):
        # type: (typing.Iterable[typing.Iterable[float]]) -> Any
        for vec in x_test:
            yield self._impl.predict_one(vec)

    def fit(self, x_train):
        # type: (numpy.array) -> None
        assert len(x_train.shape) == 2
        self._impl.fit(x_train)

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
