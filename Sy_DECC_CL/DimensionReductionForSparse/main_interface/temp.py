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

    # One
    # for func_num in [1,2,3,6,10]:
    #     name = 'f' + str(func_num)
    #     groups_One = CCDE(Dim)
    #     benchmark_summary = bench.get_info(func_num)
    #     scale_range = [benchmark_summary['lower'], benchmark_summary['upper']]
    #     f.CC_exe(Dim, func_num, NIND, int(EFs / (NIND * Dim)) - 2, scale_range, groups_One, 'One')
    #     print('Finish ', name)
    #
    # # DG
    # for func_num in [4,9,11,12,14]:
    #     name = 'f' + str(func_num)
    #     groups_DECC_DG, DECC_DG_cost = DECC_DG(func_num)
    #     benchmark_summary = bench.get_info(func_num)
    #     scale_range = [benchmark_summary['lower'], benchmark_summary['upper']]
    #     f.CC_exe(Dim, func_num, NIND, int((EFs - DECC_DG_cost) / (NIND * Dim)) - 2, scale_range, groups_DECC_DG,
    #              'DECC_DG')
    #     print('Finish ', name)

    # L
    for func_num in [5,7,8,13,15]:
        name = 'f' + str(func_num)
        groups_LASSO, LASSO_cost = LASSOCC(func_num)
        benchmark_summary = bench.get_info(func_num)
        scale_range = [benchmark_summary['lower'], benchmark_summary['upper']]
        f.CC_exe(Dim, func_num, NIND, int((EFs - LASSO_cost) / (NIND * Dim)), scale_range, groups_LASSO, 'DECC_L')
        print('Finish ', name)



