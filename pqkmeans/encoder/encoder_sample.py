import _pqkmeans
import typing
from .encoder_base import EncoderBase


class EncoderSample(EncoderBase):
    def __init__(self):
        self._impl = _pqkmeans.EncoderSample()

    def fit_generator(self, x_train):
        # type: (typing.Iterable[typing.Iterable[float]]) -> None
        self._impl.fit_generator(x_train)

    def transform_generator(self, x_test):
        # type: (typing.Iterable[typing.Iterator[float]]) -> Any
        for vector in x_test:
            yield self._impl.transform_one(vector)

    def inverse_transform_generator(self, x_test):
        # type: (typing.Iterable[typing.Iterator[int]]) -> Any
        for vector in x_test:
            yield self._impl.inverse_transform_one(vector)
