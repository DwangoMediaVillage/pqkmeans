import typing

import numpy
import sklearn
import _pqkmeans


class NativeClusteringSample(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin, sklearn.base.ClusterMixin):
    def __init__(self):
        super().__init__()
        self._impl = _pqkmeans.NativeClusteringSample()

    def fit_generator(self, x_train: typing.Iterable[typing.Iterable[float]]):
        for vec in x_train:
            if self.min_vec is None:
                self.min_vec = vec
                self.ones = numpy.ones(vec.shape, vec.dtype)
            if self.max_vec is None:
                self.max_vec = vec

            if self.ones.dot(vec) < self.ones.dot(self.min_vec):
                self.min_vec = vec
            if self.ones.dot(vec) > self.ones.dot(self.max_vec):
                self.max_vec = vec

    def transform_generator(self, x_test: typing.Iterable[typing.Iterable[float]]):
        for vec in x_test:
            if self.distance(vec, self.min_vec) < self.distance(vec, self.max_vec):
                yield 0
            else:
                yield 1

    def fit(self, x_train: numpy.array):
        assert len(x_train.shape) == 2
        self.fit_generator(x_train)

    def transform(self, x_test: numpy.array):
        assert len(x_test.shape) == 2
        return numpy.array(list(self.transform_generator(x_test)))
