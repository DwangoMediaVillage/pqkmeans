import sklearn
import _pqkmeans


class EncoderSample(sklearn.base.BaseEstimator):
    def __init__(self):
        self._impl = _pqkmeans.EncoderSample()

    def fit_generator(self, x_train):
        self._impl.fit_generator(x_train)

    def fit(self, x_train):
        self.fit_generator(x_train)

    def transform(self, x_test):
        return list(self.transform_generator(x_test))

    def transform_generator(self, x_test):
        for vector in x_test:
            yield self._impl.transform_one(vector)
