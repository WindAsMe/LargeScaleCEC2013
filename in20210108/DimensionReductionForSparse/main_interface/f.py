from in20210108.DimensionReductionForSparse.util import help
from in20210108.DimensionReductionForSparse.DE import DE
from cec2013lsgo.cec2013 import Benchmark
import numpy as np


def DECC_CL_exe(Dim, func_num, NIND, m1, scale_range, groups_One, groups_Lasso, Lasso_cost, method):
    bench = Benchmark()
    function = bench.get_function(func_num)
    EFs = 3000000
    name = 'f' + str(func_num)
    print(name, 'Optimization with', method)

    """The next is DE optimization"""
    best_indexes, best_obj_trace, Population, real_iteration = DE.DECC_CL_CCDE(Dim, NIND, m1, function, scale_range, groups_One)
    help.write_CCDE_trace(name, method, best_obj_trace)
    central_point = best_indexes[len(best_indexes)-1]
    up = [scale_range[1]] * Dim
    down = [scale_range[0]] * Dim

    for i in range(Dim):
        up[i] = max(Population[i].Chrom[:, 0])
        down[i] = min(Population[i].Chrom[:, 0])

    best_indexes, best_obj_trace = DE.DECC_CL_DECC_L(Dim, NIND, int((EFs-Lasso_cost-NIND*Dim*real_iteration)/(NIND*Dim)),
                                                     function, up, down, groups_Lasso, central_point)

    help.write_DECC_CL_trace(name, method, best_obj_trace)


def CC_exe(Dim, func_num, NIND, Max_iteration, scale_range, groups, method):
    bench = Benchmark()
    function = bench.get_function(func_num)
    name = 'f' + str(func_num)
    print(name, 'Optimization with', method)

    """The next is DE optimization"""
    best_indexes, best_obj_trace = DE.CC(Dim, NIND, Max_iteration, function, scale_range, groups)
    x = np.linspace(0, 3000000, len(best_obj_trace))
    help.draw_check(x, best_obj_trace, method)
    help.write_obj_trace(name, method, best_obj_trace)

