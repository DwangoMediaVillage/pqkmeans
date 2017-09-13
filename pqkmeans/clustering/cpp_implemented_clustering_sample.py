import typing

import numpy
import sklearn
import _pqkmeans


class CppImplementedClusteringSample(sklearn.base.BaseEstimator, sklearn.base.ClusterMixin):
    def __init__(self):
        super().__init__()
        self._impl = _pqkmeans.CppImplementedClusteringSample()

    def fit_generator(self, x_train):
        # type: (typing.Iterable[typing.Iterator[float]]) -> Any
        for vec in x_train:
            self._impl.fit_one(vec)

    def predict_generator(self, x_test):
        # type: (typing.Iterable[typing.Iterator[float]]) -> Any
        for vec in x_test:
            yield self._impl.predict_one(vec)

    def fit(self, x_train):
        # type: (numpy.array) -> None
        assert len(x_train.shape) == 2
        self.fit_generator(x_train)

    def predict(self, x_test):
        # type: (numpy.array) -> Any
        assert len(x_test.shape) == 2
        return numpy.array(list(self.predict_generator(x_test)))
