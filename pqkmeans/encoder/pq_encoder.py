import typing
import numpy
from scipy.cluster.vq import vq, kmeans2
from .encoder_base import EncoderBase


class PQEncoder(EncoderBase):
    def __init__(self, iteration=20, num_subdim=4, Ks=256):
        # type: (int, int, int) -> None
        assert Ks <= 2 ** 32
        self.iteration = iteration
        self.M, self.Ks, self.Ds = num_subdim, Ks, None
        self.code_dtype = numpy.uint8 if Ks <= 2 ** 8 else (numpy.uint16 if Ks <= 2 ** 16 else numpy.uint32)
        self.trained_encoder = None  # type: PQEncoder.TrainedPQEncoder

    def fit(self, x_train):
        # type: (numpy.array) -> None
        assert x_train.ndim == 2
        N, D = x_train.shape
        assert self.Ks < N, "the number of training vector should be more than Ks"
        assert D % self.M == 0, "input dimension must be dividable by M"
        self.Ds = int(D / self.M)
        assert self.trained_encoder is None, "fit must be called only once"

        codewords = numpy.zeros((self.M, self.Ks, self.Ds), dtype=float)
        for m in range(self.M):
            x_train_sub = x_train[:, m * self.Ds: (m + 1) * self.Ds].astype(float)
            codewords[m], _ = kmeans2(x_train_sub, self.Ks, iter=self.iteration, minit='points')
        self.trained_encoder = TrainedPQEncoder(codewords, self.code_dtype)

    def transform_generator(self, x_test):
        # type: (typing.Iterable[typing.Iterator[float]]) -> Any
        assert self.trained_encoder is not None, "This PQEncoder instance is not fitted yet. Call 'fit' with appropriate arguments before using thie method."
        return self._buffered_process(x_test, self.trained_encoder.encode_multi)

    def inverse_transform_generator(self, x_test):
        # type: (typing.Iterable[typing.Iterator[int]]) -> Any
        assert self.trained_encoder is not None, "This PQEncoder instance is not fitted yet. Call 'fit' with appropriate arguments before using thie method."
        return self._buffered_process(x_test, self.trained_encoder.decode_multi)

    @property
    def codewords(self):
        assert self.trained_encoder is not None, "This PQEncoder instance is not fitted yet. Call 'fit' with appropriate arguments before using this method."
        return self.trained_encoder.codewords


class TrainedPQEncoder(object):
    def __init__(self, codewords, code_dtype):
        # type: (numpy.array, type) -> None
        self.codewords, self.code_dtype = codewords, code_dtype
        self.M, _, self.Ds = codewords.shape
        
    def encode_multi(self, data_matrix):
        data_matrix = numpy.array(data_matrix)
        N, D = data_matrix.shape
        assert self.Ds * self.M == D, "input dimension must be Ds * M"
        
        codes = numpy.empty((N, self.M), dtype=self.code_dtype)
        for m in range(self.M):
            codes[:, m], _ = vq(data_matrix[:, m * self.Ds: (m + 1) * self.Ds], self.codewords[m])
        return codes
    
    def decode_multi(self, codes):
        codes = numpy.array(codes)
        N, M = codes.shape
        assert M == self.M
        assert codes.dtype == self.code_dtype
        
        decoded = numpy.empty((N, self.Ds * self.M), dtype=float)
        for m in range(self.M):
            decoded[:, m * self.Ds: (m + 1) * self.Ds] = self.codewords[m][codes[:, m], :]
        return decoded
