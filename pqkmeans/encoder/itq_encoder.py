import typing
import numpy
import logging
from .encoder_base import EncoderBase


class ITQEncoder(EncoderBase):
    class TrainedITQEncoder(object):
        def __init__(self, R, mean, pca_mat, bits):
            self.R, self.mean, self.pca_mat, self.bits = R, mean, pca_mat, bits

        def encode(self, vector):
            vector_pca = numpy.dot(vector - self.mean, self.pca_mat)[:self.bits]
            vector_projection = vector_pca.dot(self.R)
            vector_bin = numpy.ones(self.bits, dtype=int)
            vector_bin[vector_projection < 0] = 0
            return vector_bin

        def encode_multi(self, data_matrix):
            data_matrix_pca = numpy.dot(data_matrix - self.mean, self.pca_mat)[:, :self.bits]
            data_matrix_projection = data_matrix_pca.dot(self.R)
            data_matrix_bin = numpy.ones(data_matrix_projection.shape, dtype=int)
            data_matrix_bin[data_matrix_projection < 0] = 0
            return data_matrix_bin

        def decode(self):
            raise ("cannot decode binary to original with ITQ")

    def __init__(self, iteration: int = 50, num_bit: int = 32):
        self.iteration = iteration
        self.num_bit = num_bit
        self.trained_encoder = None

    def __preprocess(self, data, bits):
        # step0: center
        mean = numpy.mean(data, axis=0)
        data_centered = data - mean

        # step1: PCA
        cov = data_centered.T.dot(data_centered)
        l, pc = numpy.linalg.eig(cov)
        data_pca = data_centered.dot(pc)[:, :bits]

        return data_pca, mean, pc
        pass

    def __fit(self, data, bits, iteration):
        # initialize rotation randomly
        R_raw = numpy.random.rand(bits, bits)
        R, _S, _V = numpy.linalg.svd(R_raw, full_matrices=True, compute_uv=True)
        V = data

        for i in range(iteration):
            ## projection step
            print("R:", R.shape)
            print("V:", V.shape)
            VR = V.dot(R)

            ## binary assignment
            B = numpy.ones(VR.shape)
            B[VR < 0] = -1

            ## error
            error = numpy.sum((B - VR) * (B - VR))
            print("error: {}".format(error))
            error = B.T.dot(V)

            # update
            # minimize whole_error
            # https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
            UB, sigma, UA = numpy.linalg.svd(error)
            R = UB.dot(UA)
        return R

    def fit(self, x_train: numpy.array):
        assert len(x_train.shape) == 2
        assert x_train.shape[1] >= self.num_bit, "target dimention should be larger than input dimention"
        data_preprocessed, mean, pca_mat = self.__preprocess(x_train, self.num_bit)
        R = self.__fit(data_preprocessed, self.num_bit, self.iteration)
        self.trained_encoder = self.TrainedITQEncoder(R, mean, pca_mat, self.num_bit)

    def transform_generator(self, x_test: typing.Iterable[typing.Iterator[float]]):
        pass

    def inverse_transform_generator(self, x_test: typing.Iterable[typing.Iterator[int]]):
        pass


def load_huge_csv(filename, dtype=numpy.float32):
    with open(filename) as f:
        result = None
        buf = []
        for i, line in enumerate(f):
            buf.append(map(dtype, line.rstrip().split(",")))
            if i % 10000 == 1:
                print("load {}".format(i))
                if result is None:
                    result = numpy.array(buf)
                else:
                    result = numpy.vstack((result, buf))
                buf = []
        else:
            result = numpy.vstack((result, buf))
    return result


class ITQ(object):
    def train(self, data, bits, iteration=50):
        """
        :param data: numpy.array
        :param bits: int
        """
        data_preprocessed, mean, pca_mat = self.__preprocess(data, bits)
        R = self.__iteration(data_preprocessed, bits, iteration)
        return ITQEncoder(R, mean, pca_mat, bits)

    def __preprocess(self, data, bits):
        # step0: center
        mean = numpy.mean(data, axis=0)
        data_centered = data - mean

        # step1: PCA
        cov = data_centered.T.dot(data_centered)
        l, pc = numpy.linalg.eig(cov)
        data_pca = data_centered.dot(pc)[:, :bits]

        return data_pca, mean, pc

    def __iteration(self, data, bits, iteration):
        # initialize rotation randomly
        R_raw = numpy.random.rand(bits, bits)
        R, _S, _V = numpy.linalg.svd(R_raw, full_matrices=1, compute_uv=1)
        V = data

        for i in range(iteration):
            ## projection step
            logging.debug("R:", R.shape)
            logging.debug("V:", V.shape)
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
