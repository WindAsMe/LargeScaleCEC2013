from in20201127.DimensionReductionForSparse.main_interface.f import f
from in20201127.DimensionReductionForSparse.util import help
from in20201127.DimensionReductionForSparse.main_interface.grouping_main import LASSOCC, DECC_DG, CCEA
from cec2013lsgo.cec2013 import Benchmark


if __name__ == '__main__':
    # func_num = 11
    for func_num in range(4, 12):
        test_time = 25
        for i in range(test_time):
            Dim = 1000
            MAX_iteration = 100
            NIND = 30
            bench = Benchmark()
            benchmark_summary = bench.get_info(func_num)

            scale_range = [benchmark_summary['lower'], benchmark_summary['upper']]

            groups_LASSO = LASSOCC(func_num)
            for g in groups_LASSO:
                g.sort()

            f(Dim, func_num, NIND, MAX_iteration, scale_range, groups_LASSO, 'LASSO')
            # f(Dim, func_num, NIND, MAX_iteration, scale_range, groups_Normal, 'Normal')
            # f(Dim, func_num, NIND, MAX_iteration, scale_range, groups_One, 'One')
            # f(Dim, func_num, NIND, MAX_iteration, scale_range, groups_Random, 'Random')
