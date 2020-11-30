from in20201127.DimensionReductionForSparse.main_interface.optimization_f.interface_combination import f
from in20201127.DimensionReductionForSparse.util import help
from in20201127.DimensionReductionForSparse.main_interface.grouping_main import group_strategy
from cec2013lsgo.cec2013 import Benchmark


if __name__ == '__main__':

    func_num = 11
    for func_num in range(11, 3, -1):
        test_time = 10
        for i in range(test_time):
            Dim = 1000
            MAX_iteration = 100
            NIND = 29
            bench = Benchmark()
            benchmark_summary = bench.get_info(func_num)

            scale_range = [benchmark_summary['lower'], benchmark_summary['upper']]

            groups_LASSO = group_strategy(func_num)
            print('Grouping over: ', help.check_proper(groups_LASSO))
            for g in groups_LASSO:
                g.sort()

            f(Dim, func_num, NIND, MAX_iteration, scale_range, groups_LASSO, 'LASSO')
            print('    Finished: ', 'function: ', func_num, 'iteration: ', i+1, '/', test_time)

