import sklearn
import _pqkmeans
import numpy
import typing


class EncoderSample(sklearn.base.BaseEstimator):
    def __init__(self):
        self._impl = _pqkmeans.EncoderSample()

    def fit_generator(self, x_train: typing.Iterable[typing.Iterable[float]]):
        self._impl.fit_generator(x_train)

    def fit(self, x_train: numpy.array):
        self.fit_generator(x_train)

    def transform(self, x_test: numpy.array):
        return list(self.transform_generator(x_test))

    def inverse_transform(self, x_test: numpy.array):
        return list(self.inverse_transform_generator(x_test))

    def transform_generator(self, x_test: typing.Iterable[typing.Iterator[float]]):
        for vector in x_test:
            yield self._impl.transform_one(vector)

    def inverse_transform_generator(self, x_test: typing.Iterable[typing.Iterator[int]]):
        for vector in x_test:
            yield self._impl.inverse_transform_one(vector)
