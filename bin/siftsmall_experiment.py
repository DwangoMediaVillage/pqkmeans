import six
import six.moves.urllib
import os
import sys
sys.path.append(os.path.join(os.path.basename(os.path.dirname(os.path.abspath(__file__))), ".."))
import pqkmeans
import tarfile
import texmex_python
import sklearn.cluster
import numpy


def get_siftsmall():
    url = "ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz"
    filename = "siftsmall.tar.gz"
    directory = "."
    member_name = "siftsmall/siftsmall_learn.fvecs"
    path = os.path.join(directory, filename)
    if not os.path.exists(path):
        print("downloading {}".format(url))
        six.moves.urllib.request.urlretrieve(url, path)

    tardir = tarfile.open(path, "r:gz")
    member = tardir.getmember(member_name)
    data = texmex_python.reader.read_fvec(tardir.extractfile(member))
    print(data.shape)
    return data


N_CLASSES = 5
data = get_siftsmall()
kmeans = sklearn.cluster.KMeans(n_clusters=N_CLASSES)
predicted = kmeans.fit_predict(data)
total_error, micro_average_error, macro_average_error = pqkmeans.evaluation.calc_error(predicted, data, N_CLASSES)
print("""
{}:
    micro_average_error: {}
    macro_average_error: {}
""".format("Kmeans", micro_average_error, macro_average_error))


encoder = pqkmeans.encoder.ITQEncoder(num_bit=32)
encoder.fit(data)
coded = encoder.transform(data)
kmeans = pqkmeans.clustering.BKMeans(k=N_CLASSES, input_dim=32, subspace_dim=8, iteration=10)
predicted = kmeans.fit_predict(coded)
total_error, micro_average_error, macro_average_error = pqkmeans.evaluation.calc_error(predicted, data, N_CLASSES)
print("""
{}:
    micro_average_error: {}
    macro_average_error: {}
""".format("BKmeans", micro_average_error, macro_average_error))


predicted = [numpy.random.randint(0, N_CLASSES-1) for _ in range(len(predicted))]
total_error, micro_average_error, macro_average_error = pqkmeans.evaluation.calc_error(predicted, data, N_CLASSES)
print("""
{}:
    micro_average_error: {}
    macro_average_error: {}
""".format("Random", micro_average_error, macro_average_error))
