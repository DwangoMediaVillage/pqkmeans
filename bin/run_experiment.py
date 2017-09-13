import os
import sys

sys.path.append(os.path.join(os.path.basename(os.path.dirname(os.path.abspath(__file__))), ".."))
import pqkmeans
import sklearn.cluster
import numpy
import argparse

ALL_ALGORITHMS = ["kmeans", "pqkmeans", "bkmeans", "random"]

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="siftsmall", choices=["siftsmall", "sift1m", "artificial"])
parser.add_argument("--algorithms", default=ALL_ALGORITHMS, nargs="+", choices=ALL_ALGORITHMS)
parser.add_argument("--k", default=100, type=int)
args = parser.parse_args()

if args.dataset == "siftsmall":
    learn_data, test_data = pqkmeans.evaluation.get_siftsmall_dataset()
elif args.dataset == "sift1m":
    learn_data, test_data = pqkmeans.evaluation.get_sift1m_dataset()
elif args.dataset == "artificial":
    learn_data, test_data = pqkmeans.evaluation.get_gmm_random_dataset(k=args.k)
else:
    raise Exception("no such dataset: {}".format(args.dataset))

for algorithm in args.algorithms:
    if algorithm == "kmeans":
        kmeans = sklearn.cluster.KMeans(n_clusters=args.k)
        predicted = kmeans.fit_predict(test_data)
    elif algorithm == "pqkmeans":
        encoder = pqkmeans.encoder.PQEncoder()
        encoder.fit(learn_data)
        coded = encoder.transform(test_data)
        kmeans = pqkmeans.clustering.PQKMeans(k=args.k, iteration=10, encoder=encoder)
        predicted = kmeans.fit_predict(coded)
    elif algorithm == "bkmeans":
        encoder = pqkmeans.encoder.ITQEncoder(num_bit=32)
        encoder.fit(learn_data)
        coded = encoder.transform(test_data)
        kmeans = pqkmeans.clustering.BKMeans(k=args.k, input_dim=32, subspace_dim=8, iteration=10)
        predicted = kmeans.fit_predict(coded)
    elif algorithm == "random":
        predicted = [numpy.random.randint(0, args.k - 1) for _ in range(len(test_data))]
    else:
        raise Exception("no such algorithm: {}".format(algorithm))

    total_error, micro_average_error, macro_average_error = pqkmeans.evaluation.calc_error(predicted, test_data,
                                                                                           args.k)
    print("""
    {}:
        Error: {}
    """.format(algorithm, micro_average_error, macro_average_error))
