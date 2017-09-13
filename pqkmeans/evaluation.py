import collections
import numpy


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
