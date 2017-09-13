import numpy
import sklearn
import typing


class EncoderBase(sklearn.base.BaseEstimator):
    def fit_generator(self, x_train):
        # type: (typing.Iterable[typing.Iterator[float]]) -> None
        raise NotImplementedError()

    def transform_generator(self, x_test):
        # type: (typing.Iterable[typing.Iterator[float]]) -> Any
        raise NotImplementedError()

    def inverse_transform_generator(self, x_test):
        # type: (typing.Iterable[typing.Iterator[Any]]) -> Any
        raise NotImplementedError()

    def fit(self, x_train):
        # type: (numpy.array) -> None
        assert len(x_train.shape) == 2
        self.fit_generator(iter(x_train))

    def transform(self, x_test):
        # type: (numpy.array) -> Any
        assert len(x_test.shape) == 2
        return numpy.array(list(self.transform_generator(x_test)))

    def inverse_transform(self, x_test):
        # type: (numpy.array) -> Any
        assert len(x_test.shape) == 2
        return numpy.array(list(self.inverse_transform_generator(x_test)))

    def _buffered_process(self, x_input, process, buffer_size=10000):
        # type: (typing.Iterable[typing.Iterator[Any]], Any, int) -> Any
        buffer = []
        for input_vector in x_input:
            buffer.append(input_vector)
            if len(buffer) == buffer_size:
                encoded = process(buffer)
                for encoded_vec in encoded:
                    yield encoded_vec
                buffer = []
        if len(buffer) > 0:  # rest
            encoded = process(buffer)
            for encoded_vec in encoded:
                yield encoded_vec
