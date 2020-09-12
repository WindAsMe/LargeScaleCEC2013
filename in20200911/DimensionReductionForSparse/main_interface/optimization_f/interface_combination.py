from in20200911.DimensionReductionForSparse.util import help
from in20200911.DimensionReductionForSparse.DE import DE
from cec2013lsgo.cec2013 import Benchmark
import numpy as np


def f(Dim, func_num, NIND, MAX_iteration, scale_range, groups, initial_population, method):

    bench = Benchmark()
    function = bench.get_function(func_num)

    name = 'f' + str(func_num)
    print(name, 'Optimization with', method)

    """The next is DE optimization"""
    # Why in first generation has the gap?
    # Because the Grouping_0.1 strategy firstly do the best features combination in initial population

    best_obj_trace, best_index = DE.ProblemsOptimization(Dim, NIND, MAX_iteration, function, scale_range,
                                                                     groups, initial_population)
    help.write_obj_trace(name, method, best_obj_trace)
    x = np.linspace(60000, 3000000, MAX_iteration)
    help.draw_obj(x, best_obj_trace, method + 'Grouping', 'temp')
