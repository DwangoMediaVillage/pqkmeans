import collections
import numpy
import os
import six.moves.urllib
import tarfile


def get_gmm_random_dataset(k, dimension=100, test_size=5000, train_size=500):
    def random_gmm(k, n_sample):
        result = numpy.zeros((n_sample, dimension))
        for _ in range(k):
            cov_source = numpy.random.random((dimension, dimension))
            cov = cov_source.dot(cov_source.T)
            result += numpy.random.multivariate_normal(numpy.random.random(dimension), cov, n_sample)
        return result

    train_test = random_gmm(k, train_size + test_size)
    train = train_test[:train_size, :]
    test = train_test[train_size:, :]
    return train, test


def get_siftsmall_dataset(cache_directory="."):
    return get_texmex_dataset(
        url="ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz",
        filename="siftsmall.tar.gz",
        member_names=["siftsmall/siftsmall_learn.fvecs", "siftsmall/siftsmall_base.fvecs"],
        cache_directory=cache_directory,
    )

def get_sift1m_dataset(cache_directory="."):
    return get_texmex_dataset(
        url="ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz",
        filename="sift.tar.gz",
        member_names=["sift/sift_learn.fvecs", "sift/sift_base.fvecs"],
        cache_directory=cache_directory,
    )


def get_texmex_dataset(url, filename, member_names, cache_directory="."):
    try:
        import texmex_python
    except ImportError:
        raise ImportError("Missing optional dependency 'texmex_python'. You must install it to use this dataset.")
    path = os.path.join(cache_directory, filename)
    if not os.path.exists(path):
        print("downloading {}".format(url))
        six.moves.urllib.request.urlretrieve(url, path)

    learn_base = []
    for member_name in member_names:
        tardir = tarfile.open(path, "r:gz")
        member = tardir.getmember(member_name)
        data = texmex_python.reader.read_fvec(tardir.extractfile(member))
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
