import typing
import numpy
import sklearn

import _pqkmeans


class BKMeans(sklearn.base.BaseEstimator, sklearn.base.ClusterMixin):
    def __init__(self, input_dim, subspace_dim):
        super().__init__()
        self._impl = _pqkmeans.BKMeans(input_dim, subspace_dim)

    def fit_generator(self, x_train: typing.Iterable[typing.Iterable[float]]):
        for vec in x_train:
            self._impl.fit_one(vec)

    def predict_generator(self, x_test: typing.Iterable[typing.Iterable[float]]):
        for vec in x_test:
            yield self._impl.predict_one(vec)

    def fit(self, x_train: numpy.array):
        assert len(x_train.shape) == 2
        self.fit_generator(x_train)

    def predict(self, x_test: numpy.array):
        assert len(x_test.shape) == 2
        return numpy.array(list(self.predict_generator(x_test)))
