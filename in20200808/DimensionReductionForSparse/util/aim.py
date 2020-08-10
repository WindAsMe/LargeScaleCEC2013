import numpy as np
from in20200808.DimensionReductionForSparse.parameters import f_num


def fitness_evaluation(data, f, Dim, group=None):
    result = []
    if group is not None:
        for i in range(Dim):
            if i not in group:
                data[:, i] = 0

    for d in data:
        result.append([f(d, f_num)])

    return np.array(result)
