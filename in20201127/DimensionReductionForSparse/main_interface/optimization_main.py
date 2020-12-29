from in20201127.DimensionReductionForSparse.main_interface.f import f
from in20201127.DimensionReductionForSparse.util import help
from in20201127.DimensionReductionForSparse.main_interface.grouping_main import LASSOCC, DECC_DG, DECC_D, DECC_G, CCEA
from cec2013lsgo.cec2013 import Benchmark


if __name__ == '__main__':
    # func_num = 11

    LASSO_total_cost = 0
    DECC_DG_total_cost = 0
    EFs = 3000000
    for func_num in [8]:
        test_time = 25
        for i in range(test_time):
            Dim = 1000
            MAX_iteration = 100
            NIND = 30
            bench = Benchmark()
            benchmark_summary = bench.get_info(func_num)

            scale_range = [benchmark_summary['lower'], benchmark_summary['upper']]

            groups_LASSO, LASSO_cost = LASSOCC(func_num)
            LASSO_total_cost += LASSO_cost
            for g in groups_LASSO:
                g.sort()
            # DECC-G == Random
            # groups_DECC_G = DECC_G(Dim, 10, 100)
            # # cost == 100000
            # groups_DECC_D = DECC_D(func_num, 10, 100)
            # groups_DECC_DG, DECC_DG_cost = DECC_DG(func_num)
            # DECC_DG_total_cost += DECC_DG_cost

            # print('LASSOCC: ', help.check_proper(groups_LASSO), 'iter: ', int((EFs-LASSO_cost)/(NIND*Dim)))
            # print('DECC_G: ', help.check_proper(groups_DECC_G))
            # print('DECC_D: ', help.check_proper(groups_DECC_D))
            # print('DECC_DG: ', help.check_proper(groups_DECC_DG), 'iter: ', int((EFs-DECC_DG_cost)/(NIND*Dim)))

            # f(Dim, func_num, NIND, int((EFs-LASSO_cost)/(NIND*Dim)), scale_range, groups_LASSO, 'LASSO')
            # f(Dim, func_num, NIND, MAX_iteration, scale_range, groups_DECC_G, 'DECC_G')
            # f(Dim, func_num, NIND, int((EFs-100000)/(NIND*Dim)), scale_range, groups_DECC_D, 'DECC_D')
            # f(Dim, func_num, NIND, int((EFs-DECC_DG_cost)/(NIND*Dim)), scale_range, groups_DECC_DG, 'DECC_DG')

            print('    Finished: ', 'function: ', func_num, 'iteration: ', i + 1, '/', test_time)

        # help.write_cost('LASSOCC', LASSO_total_cost / test_time)
        # help.write_cost('DECC_DG', DECC_DG_total_cost / test_time)
