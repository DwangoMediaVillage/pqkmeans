import collections
import numpy
import os
import six.moves.urllib
import tarfile
import texmex_python

def get_gmm_random_dataset(dimention=100, test_size=5000, train_size=500):
    def random_gmm(k, n_sample):
        result = numpy.zeros((n_sample, dimention))
        for _ in range(k):
            cov_source = numpy.random.random((dimention, dimention))
            cov = cov_source.dot(cov_source.T)
            result += numpy.random.multivariate_normal(numpy.random.random(dimention), cov, n_sample)
        return result

    train_test = random_gmm(5, train_size+test_size)
    train = train_test[:train_size, :]
    test = train_test[train_size:, :]
    return train, test


def get_siftsmall_dataset():
    url = "ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz"
    filename = "siftsmall.tar.gz"
    directory = "."
    member_names = ["siftsmall/siftsmall_learn.fvecs", "siftsmall/siftsmall_base.fvecs"]
    path = os.path.join(directory, filename)
    if not os.path.exists(path):
        print("downloading {}".format(url))
        six.moves.urllib.request.urlretrieve(url, path)

    learn_base = []
    for member_name in member_names:
        tardir = tarfile.open(path, "r:gz")
        member = tardir.getmember(member_name)
        data = texmex_python.reader.read_fvec(tardir.extractfile(member))
        print(data.shape)
        learn_base.append(data)
    return learn_base


def calc_error(assignments, raw_features, num_classes):
    """
    calculate class internal errors
    """
    ## calculate mean feature for all classes
    mean_vectors = collections.defaultdict(lambda: None)
    count = {i: 0 for i in range(num_classes)}
    for assignment, raw_feature in zip(assignments, raw_features):
        count[assignment] += 1
        if mean_vectors[assignment] is None:
            mean_vectors[assignment] = raw_feature.copy()
        else:
            mean_vectors[assignment] += raw_feature

    mean_vectors = {
        i: sum_vector / count[i]
        for i, sum_vector in mean_vectors.items()
        }

    ## calculate sum error
    sum_errors = {i: 0 for i in range(num_classes)}
    for assignment, raw_feature in zip(assignments, raw_features):
        sum_errors[assignment] += numpy.linalg.norm(raw_feature - mean_vectors[assignment])

    ## output
    total_error = sum(sum_errors.values())
    micro_average_error = sum(sum_errors.values()) / len(assignments)
    macro_average_error = sum([
                                  sum_error / count[class_index] if count[class_index] > 0 else 0
                                  for class_index, sum_error in sum_errors.items()
                                  ]) / len(sum_errors)
    return total_error, micro_average_error, macro_average_error
