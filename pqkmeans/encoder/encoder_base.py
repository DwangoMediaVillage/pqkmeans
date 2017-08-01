import numpy
import sklearn
import typing


class EncoderBase(sklearn.base.BaseEstimator):
    def fit_generator(self, x_train: typing.Iterable[typing.Iterable[float]]):
        raise NotImplementedError()

    def transform_generator(self, x_test: typing.Iterable[typing.Iterator[float]]):
        raise NotImplementedError()

    def inverse_transform_generator(self, x_test: typing.Iterable[typing.Iterator[int]]):
        raise NotImplementedError()

    def fit(self, x_train: numpy.array):
        self.fit_generator(iter(x_train))

    def transform(self, x_test: numpy.array):
        return list(self.transform_generator(x_test))

    def inverse_transform(self, x_test: numpy.array):
        return list(self.inverse_transform_generator(x_test))
