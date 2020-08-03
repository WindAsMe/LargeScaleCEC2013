import numpy as np


def fitness_evaluation(data, f, Dim, group=None):
    result = []
    if group is not None:
        for i in range(Dim):
            if i not in group:
                data[:, i] = 0

    for d in data:
        result.append([f(d)])

    return np.array(result)
