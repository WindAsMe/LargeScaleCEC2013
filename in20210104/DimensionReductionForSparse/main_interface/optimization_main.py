from in20201230.DimensionReductionForSparse.main_interface.f import f, f2
from in20201230.DimensionReductionForSparse.util import help
from in20201230.DimensionReductionForSparse.main_interface.grouping_main import LASSOCC, DECC_DG, DECC_D, DECC_G, CCDE
from cec2013lsgo.cec2013 import Benchmark


if __name__ == '__main__':
    # func_num = 11
    Dim = 1000
    MAX_iteration = 100
    NIND = 30
    bench = Benchmark()
    LASSO_total_cost = 0
    DECC_DG_total_cost = 0
    EFs = 3000000

    for func_num in [4,5,7,8,9,11]:
        test_time = 3
        for i in range(test_time):

            benchmark_summary = bench.get_info(func_num)

            scale_range = [benchmark_summary['lower'], benchmark_summary['upper']]
            groups_One = CCDE(Dim)
            m1 = 5
            groups_LASSO, LASSO_cost = LASSOCC(func_num)
            for g in groups_LASSO:
                g.sort()
            LASSO_total_cost += LASSO_cost

            f2(Dim, func_num, NIND, m1, int((EFs-LASSO_cost-(m1*NIND*Dim))/(NIND*Dim)), scale_range, groups_One, groups_LASSO, 'CCDE_acceleration')

            print('    Finished: ', 'function: ', func_num, 'iteration: ', i + 1, '/', test_time)

        # help.write_cost('LASSOCC', LASSO_total_cost / test_time)
        # help.write_cost('DECC_DG', DECC_DG_total_cost / test_time)
