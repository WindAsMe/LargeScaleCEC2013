from in20210108.DimensionReductionForSparse.util import help
from in20210108.DimensionReductionForSparse.DE import DE
from cec2013lsgo.cec2013 import Benchmark


def DECC_CL_exe(Dim, func_num, NIND, m1, scale_range, groups_One, groups_Lasso, Lasso_cost, method):
    bench = Benchmark()
    function = bench.get_function(func_num)
    EFs = 3000000
    name = 'f' + str(func_num)
    print(name, 'Optimization with', method)

    """The next is DE optimization"""
    best_indexes, best_obj_trace, Population, real_iteration = DE.DECC_CL_CCDE(Dim, NIND, m1, function, scale_range, groups_One)
    help.write_CCDE_trace(name, method, best_obj_trace)
    central_point = best_indexes[len(best_indexes)-1]
    up = [scale_range[1]] * Dim
    down = [scale_range[0]] * Dim

    for i in range(Dim):
        up[i] = max(Population[i].Chrom[:, 0])
        down[i] = min(Population[i].Chrom[:, 0])

    best_indexes, best_obj_trace = DE.DECC_CL_DECC_L(Dim, NIND, int((EFs-Lasso_cost-NIND*Dim*real_iteration)/(NIND*Dim)),
                                                     function, up, down, groups_Lasso, central_point)

    help.write_DECC_CL_trace(name, method, best_obj_trace)


def DECC_L_exe(Dim, func_num, NIND, MAX_iteration, scale_range, groups_Lasso, method):
    bench = Benchmark()
    function = bench.get_function(func_num)
    name = 'f' + str(func_num)
    print(name, 'Optimization with', method)

    """The next is DE optimization"""
    up = [scale_range[1]] * Dim
    down = [scale_range[0]] * Dim
    central_point = help.create_data(1, Dim, scale_range)[0]
    best_indexes, best_obj_trace = DE.DECC_CL_DECC_L(Dim, NIND, MAX_iteration, function, up, down, groups_Lasso, central_point)

    help.write_DECC_CL_trace(name, method, best_obj_trace)


def CC_exe(Dim, func_num, NIND, m1, scale_range, groups_One, method):
    bench = Benchmark()
    function = bench.get_function(func_num)
    name = 'f' + str(func_num)
    print(name, 'Optimization with', method)

    """The next is DE optimization"""
    best_indexes, best_obj_trace = DE.CCDE(Dim, NIND, m1, function, scale_range, groups_One)
    help.write_obj_trace(name, method, best_obj_trace)

