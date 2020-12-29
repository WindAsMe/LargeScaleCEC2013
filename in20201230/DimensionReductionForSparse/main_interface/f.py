from in20201230.DimensionReductionForSparse.util import help
from in20201230.DimensionReductionForSparse.DE import DE
from cec2013lsgo.cec2013 import Benchmark
import numpy as np


def f(Dim, func_num, NIND, MAX_iteration, scale_range, groups, method):

    bench = Benchmark()
    function = bench.get_function(func_num)

    name = 'f' + str(func_num)
    print(name, 'Optimization with', method)

    """The next is DE optimization"""
    # Why in first generation has the gap?
    # Because the Grouping_0.1 strategy firstly do the best features combination in initial population

    best_index, best_obj_trace = DE.ProblemsOptimization(Dim, NIND, MAX_iteration, function, scale_range,
                                                                     groups)
    help.write_obj_trace(name, method, best_obj_trace)
    x = np.linspace(30000, 3000000, MAX_iteration)
    help.draw_obj(x, best_obj_trace, method + ' Grouping', 'temp')

    # if method == 'LASSO':
    #     help.write_elite(best_index[len(best_index) - 1], method)


def f2(Dim, func_num, NIND, m1, m2, scale_range, groups_One, groups_Lasso, method):
    bench = Benchmark()
    function = bench.get_function(func_num)

    name = 'f' + str(func_num)
    print(name, 'Optimization with', method)

    """The next is DE optimization"""

    best_indexes, best_obj_trace = DE.ProblemsOptimization(Dim, NIND, m1, function, scale_range, groups_One)
    central_point = best_indexes[len(best_indexes-1)]
    delta = 0.2 * (scale_range[1] - scale_range[0])
    up = []
    down = []
    for e in central_point:
        up_e = e + delta
        down_e = e - delta
        if up_e >= scale_range[1]:
            up_e = scale_range[1]
        if down_e <= scale_range[0]:
            down = scale_range[0]
        up.append(up_e)
        down.append(down_e)

    best_indexes, best_obj_trace = DE.ProblemsOptimizationForCCDEAcceleration(Dim, NIND, m2, function, up, down,
                                                                              groups_Lasso)



