from in20210108.DimensionReductionForSparse.main_interface import f
from in20210108.DimensionReductionForSparse.DE.DE import DECC_CL_DECC_L
from in20210108.DimensionReductionForSparse.util import help
from in20210108.DimensionReductionForSparse.main_interface.grouping_main import LASSOCC, DECC_DG, DECC_D, DECC_G, CCDE
from cec2013lsgo.cec2013 import Benchmark


if __name__ == '__main__':
    # func_num = 11
    Dim = 1000
    NIND = 30
    bench = Benchmark()
    Max_iteration = 100  # For 1 variable
    for func_num in range(4, 12):
        test_time = 25
        for i in range(test_time):

            benchmark_summary = bench.get_info(func_num)

            scale_range = [benchmark_summary['lower'], benchmark_summary['upper']]
            groups_One = CCDE(Dim)
            m1 = 50
            groups_LASSO = DECC_G(Dim, 20, 50)
            # groups_LASSO, LASSO_cost = LASSOCC(func_num)
            # for g in groups_LASSO:
            #     g.sort()
            # #
            # f.DECC_L_exe(Dim, func_num, NIND, 100, scale_range, groups_LASSO, 'DECC_L')
            f.DECC_CL_exe(Dim, func_num, NIND, m1, scale_range, groups_One, groups_LASSO, 10000, 'DECC_CL')

            # f.CC_exe(Dim, func_num, NIND, Max_iteration, scale_range, groups_One, 'One')
            print('    Finished: ', 'function: ', func_num, 'iteration: ', i + 1, '/', test_time)


