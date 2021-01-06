from in20210106.DimensionReductionForSparse.util import help
from in20210106.DimensionReductionForSparse.DE import DE
from cec2013lsgo.cec2013 import Benchmark
import numpy as np


def DECC_CL(Dim, func_num, NIND, m1, m2, scale_range, groups_One, groups_Lasso, method):
    bench = Benchmark()
    function = bench.get_function(func_num)

    name = 'f' + str(func_num)
    print(name, 'Optimization with', method)

    """The next is DE optimization"""
    best_indexes, best_obj_trace, Chroms = DE.DECC_CL_CCDE(Dim, NIND, m1, function, scale_range, groups_One)

    help.write_CCDE_trace(name, method, best_obj_trace)
    central_point = best_indexes[len(best_indexes)-1]
    up = [scale_range[1]] * Dim
    down = [scale_range[0]] * Dim

    if method == 'DECC_CL1':
        for i in range(Dim):
            up[i] = max(Chroms[:, i])
            down[i] = min(Chroms[:, i])

    elif method == 'DECC_CL2':
        delta = 0.25 * (scale_range[1] - scale_range[0])
        for e in central_point:
            up_e = e + delta
            down_e = e - delta
            if up_e >= scale_range[1]:
                up_e = scale_range[1]
            if down_e <= scale_range[0]:
                down_e = scale_range[0]
            up.append(up_e)
            down.append(down_e)
    best_indexes, best_obj_trace = DE.DECC_CL_DECC_L(Dim, NIND, m2, function, up, down, groups_Lasso, central_point)

    help.write_obj_trace(name, method, best_obj_trace)

