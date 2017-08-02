import typing
import numpy
from scipy.cluster.vq import vq, kmeans2
from .encoder_base import EncoderBase



class PQEncoder(EncoderBase):
    class TrainedPQEncoder(object):
        def __init__(self, codewords: numpy.array, ctype: type):
            self.codewords, self.ctype = codewords, ctype
            self.M, _, self.Ds = codewords.shape

        def encode_multi(self, data_matrix):
            # Is this OK? This expands an input generator to the matrix
            data_matrix = numpy.array(data_matrix)

            N, D = data_matrix.shape
            assert self.Ds * self.M == D, "input dimension must be Ds * M"

            codes = numpy.empty((N, self.M), dtype=self.ctype)
            for m in range(self.M):
                codes[:, m], _ = vq(data_matrix[:, m * self.Ds : (m+1) * self.Ds], self.codewords[m])
            return codes

        def decode_multi(self, codes):
            # Is this OK? This expands an input generator to the matrix
            codes = numpy.array(codes)

            N, M = codes.shape
            assert M == self.M
            assert codes.dtype == self.ctype

            decoded = numpy.empty((N, self.Ds * self.M), dtype=numpy.float)
            for m in range(self.M):
                decoded[:, m * self.Ds : (m+1) * self.Ds] = self.codewords[m][codes[:, m], :]
            return decoded

    def __init__(self, num_bit: int = 32, Ks: int = 256):
        assert num_bit % 8 == 0, "num_bit must be dividable by 8"
        self.M, self.Ks, self.Ds = int(num_bit / 8), Ks, None
        self.ctype = numpy.uint8 if Ks <= 2**8 else (numpy.uint16 if Ks <= 2**16 else numpy.uint32)
        self.trained_encoder = None  # type: PQEncoder.TrainedPQEncoder

    def fit(self, x_train: numpy.array):
        assert x_train.ndim == 2
        N, D = x_train.shape
        assert self.Ks < N, "the number of training vector should be more than Ks"
        assert D % self.M == 0, "input dimension must be dividable by M"
        self.Ds = int(D / self.M)

        codewords = numpy.zeros((self.M, self.Ks, self.Ds), dtype=numpy.float)
        for m in range(self.M):
            x_train_sub = x_train[:, m * self.Ds : (m+1) * self.Ds].astype(numpy.float)
            codewords[m], _ = kmeans2(x_train_sub, self.Ks, iter=20, minit='points')
        self.trained_encoder = self.TrainedPQEncoder(codewords, self.ctype)

    def transform_generator(self, x_test: typing.Iterable[typing.Iterator[float]]):
        assert self.trained_encoder is not None, "This PQEncoder instance is not fitted yet. Call 'fit' with appropriate arguments before using thie method."
        return self._buffered_process(x_test, self.trained_encoder.encode_multi)

    def inverse_transform_generator(self, x_test: typing.Iterable[typing.Iterator[int]]):
        assert self.trained_encoder is not None, "This PQEncoder instance is not fitted yet. Call 'fit' with appropriate arguments before using thie method."
        return self._buffered_process(x_test, self.trained_encoder.decode_multi)



