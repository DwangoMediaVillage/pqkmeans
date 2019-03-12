import typing
import numpy
import logging
import sklearn.decomposition
from .encoder_base import EncoderBase


class ITQEncoder(EncoderBase):
    class TrainedITQEncoder(object):
        def __init__(self, R, pca, bits):
            self.R, self.pca, self.bits = R, pca, bits

        def encode(self, vector):
            vector_pca = self.pca.transform(vector)
            vector_projection = vector_pca.dot(self.R)
            return vector_projection >= 0

        def encode_multi(self, data_matrix):
            data_matrix_pca = self.pca.transform(data_matrix)
            data_matrix_projection = data_matrix_pca.dot(self.R)
            return data_matrix_projection >= 0

    def __init__(self, iteration=50, num_bit=32):
        # type: (int, int) -> None
        self.iteration = iteration
        self.num_bit = num_bit
        self.trained_encoder = None  # type: ITQEncoder.TrainedITQEncoder

    def __preprocess(self, data, bits):
        pca = sklearn.decomposition.PCA(n_components=bits)
        pca.fit(data)
        data_pca = pca.transform(data)
        return data_pca, pca

    def __fit(self, data, bits, iteration):
        # initialize rotation randomly
        R_raw = numpy.random.rand(bits, bits)
        R, _S, _V = numpy.linalg.svd(R_raw, full_matrices=True, compute_uv=True)
        V = data

        for i in range(iteration):
            ## projection step
            logging.debug("R: {}".format(R.shape))
            logging.debug("V: {}".format(V.shape))
            VR = V.dot(R)

            ## binary assignment
            B = numpy.ones(VR.shape)
            B[VR < 0] = -1

            ## error
            error = numpy.sum((B - VR) * (B - VR))
            logging.debug("error: {}".format(error))
            error = B.T.dot(V)

            # update
            # minimize whole_error
            # https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
            UB, sigma, UA = numpy.linalg.svd(error)
            R = UB.dot(UA)
        return R

    def fit(self, x_train):
        # type: (numpy.array) -> None
        assert len(x_train.shape) == 2
        assert x_train.shape[1] >= self.num_bit, "target dimension should be larger than input dimension"
        data_preprocessed, pca = self.__preprocess(x_train, self.num_bit)
        R = self.__fit(data_preprocessed, self.num_bit, self.iteration)
        self.trained_encoder = self.TrainedITQEncoder(R, pca, self.num_bit)

    def transform_generator(self, x_test):
        # type: (typing.Iterable[typing.Iterator[float]]) -> Any
        assert self.trained_encoder is not None, "This ITQEncoder instance is not fitted yet. Call 'fit' with appropriate arguments before using this method."
        return self._buffered_process(x_test, self.trained_encoder.encode_multi)

    def inverse_transform_generator(self, x_test):
        # type: (typing.Iterable[typing.Iterator[int]]) -> Any
        raise ("cannot decode binary to original with ITQ")
