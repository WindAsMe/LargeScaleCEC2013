from in20210106.DimensionReductionForSparse.DE import MyProblem
from in20210106.DimensionReductionForSparse.util import help
import geatpy as ea
import numpy as np
import copy


def DECC_CL_CCDE(Dim, NIND, MAX_iteration, benchmark, scale_range, groups):
    var_traces = np.zeros((MAX_iteration, Dim))
    based_population = np.zeros(Dim)
    Chroms = np.zeros((NIND, Dim))
    for i in range(len(groups)):
        var_trace, obj_trace, chrom = CCDE_initial(NIND, MAX_iteration, benchmark, scale_range, groups[i],
                                                   based_population)

        for element in groups[i]:
            var_traces[:, element] = var_trace[:, groups[i].index(element)]
            based_population[element] = var_trace[np.argmin(obj_trace[:, 1]), groups[i].index(element)]
            Chroms[:, element] = chrom[:, groups[i].index(element)]
        # x = np.linspace(0, len(groups[i]) * MAX_iteration * NIND, MAX_iteration)
        # help.draw_obj(x, obj_trace[:, 1], 'method', 'temp')

    var_traces, obj_traces = help.preserve(var_traces, benchmark)
    fit = []
    for c in Chroms:
        fit.append(benchmark(c))
    return var_traces, obj_traces, Chroms


def DECC_CL_DECC_L(Dim, NIND, MAX_iteration, benchmark, up, down, groups, elite):
    var_traces = np.zeros((MAX_iteration, Dim))
    based_population = copy.deepcopy(elite)
    for i in range(len(groups)):
        var_trace, obj_trace = Block_Optimization(NIND, MAX_iteration, benchmark, up, down, groups[i],
                                                                    based_population)

        for element in groups[i]:
            var_traces[:, element] = var_trace[:, groups[i].index(element)]
            based_population[element] = var_trace[np.argmin(obj_trace[:, 1]), groups[i].index(element)]

    var_traces[0] = copy.deepcopy(elite)
    var_traces, obj_traces = help.preserve(var_traces, benchmark)
    return var_traces, obj_traces


def Block_Optimization(NIND, MAX_iteration, benchmark, up, down, group, based_population):
    problem = MyProblem.Block_Problem(group, benchmark, up, down, NIND * len(group), based_population)  # 实例化问题对象

    """==============================种群设置==========================="""
    Encoding = 'RI'  # 编码方式
    NIND = NIND * len(group)  # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
    population = ea.Population(Encoding, Field, NIND)
    chrom = []
    for e in group:
        chrom.append(based_population[e])
    population.initChrom(NIND)
    population.Chrom[0] = chrom
    """===========================算法参数设置=========================="""

    # myAlgorithm = templet.soea_SaNSDE_templet(problem, population)
    myAlgorithm = ea.soea_DE_currentToBest_1_L_templet(problem, population)
    myAlgorithm.MAXGEN = MAX_iteration
    myAlgorithm.drawing = 0
    """=====================调用算法模板进行种群进化====================="""
    # [population, obj_trace, var_trace] = myAlgorithm.run(population, MAX_iteration)
    [population, obj_trace, var_trace] = myAlgorithm.run()
    # obj_traces.append(obj_trace[0])
    return var_trace, obj_trace


def CCDE_initial(NIND, MAX_iteration, benchmark, scale_range, group, based_population):
    problem = MyProblem.CCDE_Problem(group, benchmark, scale_range, NIND * len(group), based_population)  # 实例化问题对象

    """==============================种群设置==========================="""
    Encoding = 'RI'  # 编码方式
    NIND = NIND * len(group)  # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
    population = ea.Population(Encoding, Field, NIND)
    population.initChrom(NIND)
    """===========================算法参数设置=========================="""

    # myAlgorithm = templet.soea_SaNSDE_templet(problem, population)
    myAlgorithm = ea.soea_DE_currentToBest_1_L_templet(problem, population)
    myAlgorithm.MAXGEN = MAX_iteration
    myAlgorithm.drawing = 0
    """=====================调用算法模板进行种群进化====================="""
    # [population, obj_trace, var_trace] = myAlgorithm.run(population, MAX_iteration)
    [population, obj_trace, var_trace] = myAlgorithm.run()
    # obj_traces.append(obj_trace[0])
    return var_trace, obj_trace, population.Chrom


def OptTool(Dim, NIND, MAX_iteration, benchmark, scale_range, maxormin):
    problem = MyProblem.MyProblem(Dim, benchmark, scale_range, maxormin)  # 实例化问题对象

    """==============================种群设置==========================="""
    Encoding = 'RI'  # 编码方式
    NIND = NIND  # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
    population = ea.Population(Encoding, Field, NIND)
    population.initChrom(NIND)
    """===========================算法参数设置=========================="""

    # myAlgorithm = templet.soea_SaNSDE_templet(problem, population)
    myAlgorithm = ea.soea_DE_currentToBest_1_L_templet(problem, population)
    myAlgorithm.MAXGEN = MAX_iteration
    myAlgorithm.drawing = 0
    """=====================调用算法模板进行种群进化====================="""
    # [population, obj_trace, var_trace] = myAlgorithm.run(population, MAX_iteration)
    [population, obj_trace, var_trace] = myAlgorithm.run()
    # obj_traces.append(obj_trace[0])
    return var_trace[len(var_trace)-1]
