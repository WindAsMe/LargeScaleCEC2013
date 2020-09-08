from in20200828.DimensionReductionForSparse.util import help
from in20200828.DimensionReductionForSparse.DE import DE
from cec2013lsgo.cec2013 import Benchmark
import numpy as np


if __name__ == '__main__':
    Dim = 1000
    func_num = 14
    bench = Benchmark()
    benchmark_function = bench.get_function(func_num)
    benchmark_summary = bench.get_info(func_num)
    scale_range = [benchmark_summary['lower'], benchmark_summary['upper']]

    name = 'f' + str(func_num)
    print(name, 'Optimization')
    print('scale range: ', scale_range)
    groups_One = help.groups_one_create(Dim)
    """The next is DE optimization"""
    # Why in first generation has the gap?
    # Because the grouping strategy firstly do the best features combination in initial population
    simple_population_size = 30
    simple_MAX_iteration = 99
    test_times = 1
    max_or_min = 1

    for t in range(test_times):
        print('round', t + 1)
        best_One_obj_trace, best_One_index = DE.SimpleProblemsOptimization(Dim, simple_population_size,
                                                                               simple_MAX_iteration, benchmark_function,
                                                                               scale_range, groups_One, max_or_min)

        help.write_obj_trace(name, 'One', best_One_obj_trace)
        help.write_var_trace(name, 'One', best_One_index)
        x = np.linspace(0, 3000000, simple_MAX_iteration)
        help.draw_obj(x, best_One_obj_trace, 'One Grouping', 'temp')
