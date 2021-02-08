from Asy_DECC_CL.DimensionReductionForSparse.main_interface import f
from Asy_DECC_CL.DimensionReductionForSparse.util import help
from Asy_DECC_CL.DimensionReductionForSparse.main_interface.grouping_main import LASSOCC, DECC_DG, DECC_D, DECC_G, CCDE, Normal
from cec2013lsgo.cec2013 import Benchmark


if __name__ == '__main__':
    # func_num = 11
    Dim = 1000
    NIND = 30
    bench = Benchmark()
    EFs = 3000000
    for func_num in range(4, 12):
        test_time = 2
        for i in range(test_time):

            benchmark_summary = bench.get_info(func_num)

            scale_range = [benchmark_summary['lower'], benchmark_summary['upper']]
            groups_One = CCDE(Dim)
            # groups_DECC_G = DECC_G(Dim, 10, 100)
            # groups_DECC_D = DECC_D(func_num, 10, 100)
            # groups_DECC_DG, DECC_DG_cost = DECC_DG(func_num)
            m1 = 100
            groups_LASSO, LASSO_cost = LASSOCC(func_num)
            for g in groups_LASSO:
                g.sort()

            # f.CC_exe(Dim, func_num, NIND, int(EFs / (NIND * Dim)) - 2, scale_range, groups_One, 'One', 'Sy')
            # f.CC_exe(Dim, func_num, NIND, int(EFs / (NIND * Dim)) - 2, scale_range, groups_DECC_G, 'DECC_G')
            # f.CC_exe(Dim, func_num, NIND, int((EFs - 100000) / (NIND * Dim)) - 2, scale_range, groups_DECC_D, 'DECC_D')
            # f.CC_exe(Dim, func_num, NIND, int((EFs - DECC_DG_cost) / (NIND * Dim)) - 2, scale_range, groups_DECC_DG, 'DECC_DG')
            # f.CC_exe(Dim, func_num, NIND, int((EFs - LASSO_cost) / (NIND * Dim)), scale_range, groups_LASSO, 'DECC_L')
            f.DECC_CL_exe(Dim, func_num, NIND, m1, scale_range, groups_One, groups_LASSO, LASSO_cost, 'DECC_CL')

            print('    Finished: ', 'function: ', func_num, 'iteration: ', i + 1, '/', test_time)


