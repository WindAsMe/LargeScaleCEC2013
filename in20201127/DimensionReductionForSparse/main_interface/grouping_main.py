from in20201127.DimensionReductionForSparse.util import help
from in20201127.DimensionReductionForSparse.Sparse import SparseModel
from cec2013lsgo.cec2013 import Benchmark
import random


def group_strategy(func_num):
    Dim = 1000
    size = 5000
    degree = 3
    bench = Benchmark()
    group_dim = 50
    max_variables_num = 50


    function = bench.get_function(func_num)
    benchmark_summary = bench.get_info(func_num)
    scale_range = [benchmark_summary['lower'], benchmark_summary['upper']]

    All_groups = []
    for current_index in range(0, 20):
        Lasso_model, Feature_names = SparseModel.Regression(degree, size, Dim, group_dim, current_index, scale_range, function)

        # Grouping
        coef, Feature_names = help.not_zero_feature(Lasso_model.coef_,
                                                        help.feature_names_normalization(Feature_names))
        groups = help.group_DFS(group_dim, Feature_names, max_variables_num)

        bias = current_index * group_dim
        for g in groups:
            for i in range(len(g)):
                g[i] += bias

        for g in groups:
            if not g or g is None:
                groups.remove(g)
        # We need to check the relationship between new groups and previous groups
        temp_groups = []

        for i in range(len(All_groups)):
            for j in range(len(groups)):
                if i < len(All_groups) and j < len(groups):
                    flag = 0
                    for verify in range(3):
                        a1 = random.randint(0, len(All_groups[i])-1)
                        a2 = random.randint(0, len(groups[j])-1)
                        if not help.Differential(All_groups[i][a1], groups[j][a2], function):
                            flag += 1
                    if flag >= 1:
                        g1 = All_groups.pop(i)
                        g2 = groups.pop(j)
                        temp_groups.append(g1+g2)
                        i -= 1
                        j -= 1
                        break

        for g in All_groups:
            temp_groups.append(g)
        for g in groups:
            temp_groups.append(g)
        All_groups = temp_groups.copy()
    return All_groups
