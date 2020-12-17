from in20201127.DimensionReductionForSparse.main_interface.f import f
from in20201127.DimensionReductionForSparse.util import help
from in20201127.DimensionReductionForSparse.main_interface.grouping_main import LASSOCC, DECC_DG, DECC_D, DECC_G, CCEA
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
            groups_DECC_G = DECC_G(Dim, 10, 100)
            groups_DECC_D = DECC_D(func_num, 10, 100)
            groups_DECC_DG = DECC_DG(func_num)

            print('LASSOCC: ', help.check_proper(groups_LASSO))
            print('DECC_G: ', help.check_proper(groups_DECC_G))
            print('DECC_D: ', help.check_proper(groups_DECC_D))
            print('DECC_DG: ', help.check_proper(groups_DECC_DG))

            f(Dim, func_num, NIND, MAX_iteration, scale_range, groups_LASSO, 'LASSO')
            f(Dim, func_num, NIND, MAX_iteration, scale_range, groups_DECC_G, 'DECC_G')
            f(Dim, func_num, NIND, MAX_iteration, scale_range, groups_DECC_D, 'DECC_D')
            f(Dim, func_num, NIND, MAX_iteration, scale_range, groups_DECC_DG, 'DECC_DG')

            print('    Finished: ', 'function: ', func_num, 'iteration: ', i + 1, '/', test_time)
