from in20210108.DimensionReductionForSparse.DE import MyProblem
from in20210108.DimensionReductionForSparse.util import help
import geatpy as ea
import numpy as np
import matplotlib.pyplot as plt
import copy


# Asynchronous
def DECC_CL_CCDE(Dim, NIND, MAX_iteration, benchmark, scale_range, groups):
    var_traces = np.zeros((MAX_iteration, Dim))
    based_population = np.zeros(Dim)
    initial_Population = help.initial_population(NIND, groups, [scale_range[1]] * Dim, [scale_range[0]] * Dim, [0] * Dim)
    Best_previous = float('inf')
    Best_current = 1e50
    real_iteration = 0

    while help.is_Continue(Best_previous, Best_current, threshold=0.001) and real_iteration < MAX_iteration:
        Best_previous = Best_current
        for i in range(len(groups)):
            var_trace, obj_trace, population = CCDE_initial(NIND, 1, benchmark, scale_range, groups[i], based_population,
                                                       initial_Population[i])
            initial_Population[i] = population

            for element in groups[i]:
                var_traces[real_iteration, element] = var_trace[:, groups[i].index(element)]
                based_population[element] = var_trace[np.argmin(obj_trace[:, 1]), groups[i].index(element)]

        Best_current = benchmark(var_traces[real_iteration])
        real_iteration += 1

    var_traces = var_traces[0:real_iteration]

    var_traces, obj_traces = help.preserve(var_traces, benchmark)
    return var_traces, obj_traces, initial_Population, real_iteration


def CCDE_initial(NIND, MAX_iteration, benchmark, scale_range, group, based_population, p):
    problem = MyProblem.CCDE_Problem(group, benchmark, scale_range, NIND, based_population)  # 实例化问题对象

    """==============================种群设置==========================="""
    population = p
    """===========================算法参数设置=========================="""

    myAlgorithm = ea.soea_DE_currentToBest_1_L_templet(problem, population)
    myAlgorithm.MAXGEN = MAX_iteration
    myAlgorithm.drawing = 0
    """=====================调用算法模板进行种群进化====================="""
    # [population, obj_trace, var_trace] = myAlgorithm.run(population, MAX_iteration)
    [population, obj_trace, var_trace] = myAlgorithm.run()
    # obj_traces.append(obj_trace[0])

    return var_trace, obj_trace, population


def DECC_CL_DECC_L(Dim, NIND, MAX_iteration, benchmark, up, down, groups, elite):
    print(benchmark(elite))
    print(MAX_iteration)
    var_traces = np.zeros((MAX_iteration+1, Dim))
    var_traces[0] = copy.deepcopy(elite)

    based_population = copy.deepcopy(elite)
    initial_Population = help.initial_population(NIND, groups, up, down, elite)
    for real_iteration in range(1, MAX_iteration+1):
        for i in range(len(groups)):

            var_trace, obj_trace, population = Block_Optimization(NIND*len(groups[i]), 1, benchmark, up, down, groups[i],
                                                                    based_population, initial_Population[i])
            initial_Population[i] = population
            for element in groups[i]:
                var_traces[real_iteration, element] = var_trace[0, groups[i].index(element)]
                # based_population[element] = var_trace[0, groups[i].index(element)]

    x = np.linspace(0, 1000, len(var_traces))
    y = []
    for v in var_traces:
        y.append(benchmark(v))
    plt.plot(x, y, label='DECC-CL')

    plt.xlabel('Evaluation times')
    plt.ylabel('Fitness')
    plt.legend()
    plt.show()

    var_traces, obj_traces = help.preserve(var_traces, benchmark)
    return var_traces, obj_traces


# def DECC_L(Dim, NIND, MAX_iteration, benchmark, up, down, groups, elite):
#     var_traces = np.zeros((MAX_iteration, Dim))
#     based_population = copy.deepcopy(elite)
#     initial_Chrom = help.initial_population(NIND, groups, up, down, elite)
#     for real_iteration in range(MAX_iteration):
#         for i in range(len(groups)):
#             var_trace, obj_trace, c = Block_Optimization(NIND*len(groups[i]), 1, benchmark, up, down, groups[i],
#                                                                     based_population, initial_Chrom[i])
#             initial_Chrom[i] = c
#             for element in groups[i]:
#                 var_traces[real_iteration, element] = var_trace[0, groups[i].index(element)]
#                 based_population[element] = var_trace[0, groups[i].index(element)]
#
#     x = np.linspace(0, 1000, len(var_traces))
#     y = []
#     for v in var_traces:
#         y.append(benchmark(v))
#     plt.plot(x, y, label='DECC-L')
#
#     plt.xlabel('Evaluation times')
#     plt.ylabel('Fitness')
#     plt.legend()
#     plt.show()
#
#     if benchmark(var_traces[0]) > benchmark(elite):
#         var_traces[0] = copy.deepcopy(elite)
#     var_traces, obj_traces = help.preserve(var_traces, benchmark)
#     return var_traces, obj_traces


def Block_Optimization(NIND, MAX_iteration, benchmark, up, down, group, based_population, p):
    problem = MyProblem.Block_Problem(group, benchmark, up, down, NIND, based_population)  # 实例化问题对象

    """===========================算法参数设置=========================="""

    # myAlgorithm = templet.soea_SaNSDE_templet(problem, population)
    myAlgorithm = ea.soea_DE_currentToBest_1_L_templet(problem, p)
    myAlgorithm.MAXGEN = MAX_iteration
    myAlgorithm.drawing = 0
    """=====================调用算法模板进行种群进化====================="""
    # [population, obj_trace, var_trace] = myAlgorithm.run(population, MAX_iteration)
    [population, obj_trace, var_trace] = myAlgorithm.run()
    # obj_traces.append(obj_trace[0])
    return var_trace, obj_trace, population


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


def CCDE(Dim, NIND, MAX_iteration, benchmark, scale_range, groups):
    var_traces = np.zeros((MAX_iteration, Dim))
    based_population = np.zeros(Dim)
    Current_Chroms = np.zeros((NIND, Dim))
    real_iteration = 0

    while real_iteration < MAX_iteration:
        Previous_Chroms = copy.deepcopy(Current_Chroms)
        for i in range(len(groups)):
            var_trace, obj_trace, chrom = CCDE_initial(NIND, 1, benchmark, scale_range, groups[i], based_population,
                                                       Previous_Chroms)
            for element in groups[i]:
                var_traces[real_iteration, element] = var_trace[:, groups[i].index(element)]
                based_population[element] = var_trace[np.argmin(obj_trace[:, 1]), groups[i].index(element)]
                Current_Chroms[:, element] = chrom[:, groups[i].index(element)]

        real_iteration += 1

    var_traces = var_traces[0:real_iteration]
    var_traces, obj_traces = help.preserve(var_traces, benchmark)
    return var_traces, obj_traces
