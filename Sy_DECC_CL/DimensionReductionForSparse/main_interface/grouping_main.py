from in20210119.DimensionReductionForSparse.util import help
from in20210119.DimensionReductionForSparse.Sparse import SparseModel
from in20210119.DimensionReductionForSparse.DE import DE
from cec2013lsgo.cec2013 import Benchmark
import random
import numpy as np


def LASSOCC(func_num):
    Dim = 1000
    size = 5000
    degree = 3
    bench = Benchmark()
    group_dim = 50
    max_variables_num = 50

    function = bench.get_function(func_num)
    benchmark_summary = bench.get_info(func_num)
    scale_range = [benchmark_summary['lower'], benchmark_summary['upper']]
    verify_time = 0
    All_groups = []
    for current_index in range(0, 20):
        # print(current_index)
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

        # for g in groups:
        #     All_groups.append(g)
        # We need to check the relationship between new groups and previous groups
        temp_groups = []
        for i in range(len(All_groups)):
            for j in range(len(groups)):
                if i < len(All_groups) and j < len(groups):
                    flag = 0
                    for verify in range(3):
                        verify_time += 1
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
    # print('verify time: ', verify_time)
    return All_groups, verify_time + 100000


def CCDE(Dim):
    groups = []
    for i in range(Dim):
        groups.append([i])
    return groups


def Normal(Dim=1000):
    group = []
    for i in range(Dim):
        group.append(i)
    return [group]


def DECC_DG(func_num):
    cost = 0
    bench = Benchmark()
    function = bench.get_function(func_num)
    groups = [[0]]
    for i in range(1, 1000):
        flag = False
        for group in groups:
            for e in group:
                cost += 1
                if not help.Differential(e, i, function):
                    group.extend([i])
                    flag = True
                    break
            if flag:
                break
        if flag is False:
            groups.append([i])
    return groups, cost


def DECC_G(Dim, groups_num=10, max_number=100):
    groups = []
    for i in range(groups_num):
        groups.append([])
    for i in range(Dim):
        index = random.randint(0, groups_num-1)
        while len(groups[index]) > max_number:
            index = random.randint(0, groups_num - 1)
        groups[index].append(i)
    # print(groups)
    return groups


def DECC_D(func_num, groups_num=10, max_number=100):
    bench = Benchmark()
    function = bench.get_function(func_num)
    benchmark_summary = bench.get_info(func_num)
    scale_range = [benchmark_summary['lower'], benchmark_summary['upper']]
    Dim = 1000
    NIND = 10000
    iter = 5
    find_max = -1
    find_min = 1
    max_index = DE.OptTool(Dim, NIND, iter, function, scale_range, find_max)
    min_index = DE.OptTool(Dim, NIND, iter, function, scale_range, find_min)
    index_difference = []
    for i in range(Dim):
        index_difference.append(abs(max_index[i] - min_index[i]))
    sort_index = np.argsort(index_difference).tolist()

    groups = []
    for i in range(groups_num):
        groups.append(sort_index[i*max_number:(i+1)*max_number])
    return groups

