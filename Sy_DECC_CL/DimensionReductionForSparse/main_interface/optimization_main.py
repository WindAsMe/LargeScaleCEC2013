from Sy_DECC_CL.DimensionReductionForSparse.main_interface import f
from Sy_DECC_CL.DimensionReductionForSparse.util import help
from Sy_DECC_CL.DimensionReductionForSparse.main_interface.grouping_main import LASSOCC, DECC_DG, DECC_D, DECC_G, CCDE, Normal
from cec2013lsgo.cec2013 import Benchmark
import time


if __name__ == '__main__':
    # func_num = 11
    Dim = 1000
    NIND = 30
    bench = Benchmark()
    EFs = 3000000
    for func_num in range(1, 16):
        test_time = 1
        name = 'f' + str(func_num)
        for i in range(test_time):

            benchmark_summary = bench.get_info(func_num)

            scale_range = [benchmark_summary['lower'], benchmark_summary['upper']]

            groups_Normal = Normal(Dim)

            base_time = time.time()
            groups_One = CCDE(Dim)
            O_group_time = time.time()

            groups_DECC_G = DECC_G(Dim, 10, 100)
            G_group_time = time.time()

            groups_DECC_D = DECC_D(func_num, 10, 100)
            D_group_time = time.time()

            groups_DECC_DG, DECC_DG_cost = DECC_DG(func_num)
            DG_group_time = time.time()

            groups_LASSO, LASSO_cost = LASSOCC(func_num)
            L_group_time = time.time()

            help.write_EFS_cost(name, 'DECC_DG_EFS', str(DECC_DG_cost))
            help.write_EFS_cost(name, 'LASSO_cost_EFS', str(DECC_DG_cost))

            base_time_2 = time.time()
            f.CC_exe(Dim, func_num, NIND, int(EFs / (NIND * Dim)) - 2, scale_range, groups_Normal, 'Normal')
            N_Opt_time = time.time()

            f.CC_exe(Dim, func_num, NIND, int(EFs / (NIND * Dim)) - 2, scale_range, groups_One, 'One')
            O_Opt_time = time.time()

            f.CC_exe(Dim, func_num, NIND, int(EFs / (NIND * Dim)) - 2, scale_range, groups_DECC_G, 'DECC_G')
            G_Opt_time = time.time()

            f.CC_exe(Dim, func_num, NIND, int((EFs - 100000) / (NIND * Dim)) - 2, scale_range, groups_DECC_D, 'DECC_D')
            D_Opt_time = time.time()

            f.CC_exe(Dim, func_num, NIND, int((EFs - DECC_DG_cost) / (NIND * Dim)) - 2, scale_range, groups_DECC_DG, 'DECC_DG')
            DG_Opt_time = time.time()

            f.CC_exe(Dim, func_num, NIND, int((EFs - LASSO_cost) / (NIND * Dim)), scale_range, groups_LASSO, 'DECC_L')
            L_Opt_time = time.time()

            f.DECC_CL_exe(Dim, func_num, NIND, scale_range, groups_One, groups_LASSO, LASSO_cost, 'DECC_CL')
            CL_Opt_time = time.time()

            help.write_CPU_cost(name, 'Normal_CPU', str(N_Opt_time - base_time_2))
            help.write_CPU_cost(name, 'One_CPU', str(O_Opt_time - N_Opt_time + O_group_time - base_time))
            help.write_CPU_cost(name, 'DECC_G_CPU', str(G_Opt_time - O_Opt_time + G_group_time - O_group_time))
            help.write_CPU_cost(name, 'DECC_D_CPU', str(D_Opt_time - G_Opt_time + D_group_time - G_group_time))
            help.write_CPU_cost(name, 'DECC_DG_CPU', str(DG_Opt_time - D_Opt_time + DG_group_time - D_group_time))
            help.write_CPU_cost(name, 'DECC_L_CPU', str(L_Opt_time - DG_Opt_time + L_group_time - DG_group_time))
            help.write_CPU_cost(name, 'DECC_CL_CPU', str(CL_Opt_time - L_Opt_time + L_group_time - DG_group_time))

            print('    Finished: ', 'function: ', func_num, 'iteration: ', i + 1, '/', test_time)


