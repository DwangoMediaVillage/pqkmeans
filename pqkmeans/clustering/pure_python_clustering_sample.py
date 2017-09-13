import typing
import sklearn
import numpy


class PurePythonClusteringSample(sklearn.base.BaseEstimator, sklearn.base.ClusterMixin):
    def __init__(self):
        self.min_vec = None
        self.max_vec = None
        self.ones = None  # decision boundary

    def distance(self, x, y):
        x_projected = self.ones.dot(x)
        y_projected = self.ones.dot(y)
        return numpy.abs(x_projected - y_projected)

    def fit_generator(self, x_train):
        # type: (typing.Iterable[typing.Iterable[float]]) -> None
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

    def predict_generator(self, x_test):
        # type: (typing.Iterable[typing.Iterable[float]]) -> Any
        for vec in x_test:
            if self.distance(vec, self.min_vec) < self.distance(vec, self.max_vec):
                yield 0
            else:
                yield 1

    def fit(self, x_train):
        # type: (numpy.array) -> None
        assert len(x_train.shape) == 2
        self.fit_generator(x_train)

    def predict(self, x_test):
        # type: (numpy.array) -> Any
        assert len(x_test.shape) == 2
        return numpy.array(list(self.predict_generator(x_test)))
