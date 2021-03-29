from Sy_DECC_CL.DimensionReductionForSparse.DE import MyProblem, templet
from Sy_DECC_CL.DimensionReductionForSparse.util import help
import geatpy as ea
import numpy as np
import matplotlib.pyplot as plt
import copy


# Synchronous
def DECC_CL_CCDE(Dim, NIND, benchmark, scale_range, groups):
    var_traces = np.zeros((1, Dim))
    testify_var_traces = np.zeros((1, Dim))
    based_population = np.zeros(Dim)
    initial_Population = help.initial_population(NIND, groups, [scale_range[1]] * Dim, [scale_range[0]] * Dim, based_population)
    max_iteration = 0

    for i in range(len(groups)):
        real_iteration = 0
        Obj_traces = []
        while real_iteration < 1:
            var_trace, obj_trace, initial_Population[i] = CC_Optimization(1, benchmark, scale_range, groups[i],
                                                               based_population, initial_Population[i], real_iteration)

            index = np.argmin(obj_trace[:, 1])
            testify_var_traces[real_iteration, i] = format(var_trace[index, 0], '.8f')
            var_traces[real_iteration, i] = format(var_trace[index, 0], '.8f')
            based_population[i] = format(var_trace[index, 0], '.8f')
            Obj_traces.append(benchmark(testify_var_traces[real_iteration]))

            real_iteration += 1

        testify_var_traces[:, i] = testify_var_traces[real_iteration-1, i]

        max_iteration = max(real_iteration, max_iteration)
        for index in range(real_iteration, 1):
            var_traces[index][i] = var_traces[real_iteration-1][i]

    var_traces = var_traces[0:max_iteration]
    var_traces, obj_traces = help.preserve(var_traces, benchmark)
    return var_traces, obj_traces, initial_Population


def DECC_CL_DECC_L(Dim, NIND, MAX_iteration, benchmark, up, down, groups, elite):
    # print('elite: ', benchmark(elite))
    var_traces = np.zeros((MAX_iteration+1, Dim))
    var_traces[0] = copy.deepcopy(elite)
    based_population = copy.deepcopy(elite)
    initial_population = help.initial_population(NIND, groups, up, down, elite)

    for i in range(len(groups)):
        var_trace, obj_trace = DECC_L_Sy(MAX_iteration, benchmark, up, down, groups[i], based_population,
                                          initial_population[i])
        # x = np.linspace(0, len(obj_trace), len(obj_trace))
        # help.draw_check(x, obj_trace, 'CC')
        for element in groups[i]:
            var_traces[1:MAX_iteration+1, element] = var_trace[:, groups[i].index(element)]
            v = var_trace[len(var_trace)-1]
            based_population[element] = v[groups[i].index(element)]

    var_traces, obj_traces = help.preserve(var_traces, benchmark)

    return var_traces, obj_traces


def DECC_L_Sy(MAX_iteration, benchmark, up, down, group, based_population, initial_population):
    problem = MyProblem.Block_Problem(group, benchmark, up, down, based_population)  # 实例化问题对象


    """===========================算法参数设置=========================="""

    myAlgorithm = ea.soea_DE_currentToBest_1_L_templet(problem, initial_population)
    myAlgorithm.MAXGEN = MAX_iteration
    myAlgorithm.drawing = 0
    """=====================调用算法模板进行种群进化====================="""
    # [population, obj_trace, var_trace] = myAlgorithm.run(population, MAX_iteration)
    [population, obj_trace, var_trace] = myAlgorithm.run()
    # obj_traces.append(obj_trace[0])
    return var_trace, obj_trace[:, 1]


def OptTool(Dim, NIND, MAX_iteration, benchmark, group, scale_range, maxormin):
    problem = MyProblem.MyProblem(Dim, group, benchmark, scale_range, maxormin)  # 实例化问题对象

    """==============================种群设置==========================="""
    Encoding = 'RI'  # 编码方式
    NIND = NIND  # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
    population = ea.Population(Encoding, Field, NIND)
    population.initChrom(NIND)
    """===========================算法参数设置=========================="""
    delta_before = []
    for i in range(len(group)):
        delta_before.append(sum(population.Chrom[:, i]) / NIND)
    # myAlgorithm = templet.soea_SaNSDE_templet(problem, population)
    myAlgorithm = ea.soea_DE_currentToBest_1_L_templet(problem, population)
    myAlgorithm.MAXGEN = MAX_iteration
    myAlgorithm.drawing = 0
    """=====================调用算法模板进行种群进化====================="""
    # [population, obj_trace, var_trace] = myAlgorithm.run(population, MAX_iteration)
    [population, obj_trace, var_trace] = myAlgorithm.run()
    # obj_traces.append(obj_trace[0])
    delta_after = []
    for i in range(len(group)):
        delta_after.append(sum(population.Chrom[:, i]) / NIND)
    delta = []
    for i in range(len(group)):
        delta.append(delta_after[i] - delta_before[i])
    return delta


def CC_Asy(Dim, NIND, MAX_iteration, benchmark, scale_range, groups):
    var_traces = np.zeros((MAX_iteration, Dim))
    based_population = np.zeros(Dim)
    initial_Population = help.initial_population(NIND, groups, [scale_range[1]]*Dim, [scale_range[0]]*Dim, based_population)
    real_iteration = 0

    while real_iteration < MAX_iteration:
        for i in range(len(groups)):
            var_trace, obj_trace, population = CC_Optimization(1, benchmark, scale_range, groups[i],
                                                       based_population, initial_Population[i], real_iteration)

            initial_Population[i] = population
            for element in groups[i]:
                var_traces[real_iteration, element] = var_trace[1, groups[i].index(element)]
                based_population[element] = var_trace[1, groups[i].index(element)]
        real_iteration += 1

    var_traces, obj_traces = help.preserve(var_traces, benchmark)
    return var_traces, obj_traces


def CC_Sy(Dim, NIND, MAX_iteration, benchmark, scale_range, groups):
    var_traces = np.zeros((MAX_iteration, Dim))
    based_population = np.zeros(Dim)
    for i in range(len(groups)):
        var_trace, obj_trace = CC_Optimization_Sy(NIND, MAX_iteration, benchmark, scale_range, groups[i], based_population)

        for element in groups[i]:
            var_traces[:, element] = var_trace[:, groups[i].index(element)]
            based_population[element] = var_trace[np.argmin(obj_trace[:, 1]), groups[i].index(element)]

    var_traces, obj_traces = help.preserve(var_traces, benchmark)
    return var_traces, obj_traces


def CC_Optimization_Sy(NIND, MAX_iteration, benchmark, scale_range, group, based_population):
    problem = MyProblem.CC_Problem(group, benchmark, scale_range, based_population)  # 实例化问题对象

    """===========================算法参数设置=========================="""
    Encoding = 'RI'  # 编码方式
    NIND = NIND * len(group)  # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
    population = ea.Population(Encoding, Field, NIND)
    population.initChrom()
    myAlgorithm = ea.soea_DE_currentToBest_1_L_templet(problem, population)
    myAlgorithm.MAXGEN = MAX_iteration
    myAlgorithm.drawing = 0
    """=====================调用算法模板进行种群进化====================="""
    # [population, obj_trace, var_trace] = myAlgorithm.run(population, MAX_iteration)
    [population, obj_trace, var_trace] = myAlgorithm.run()
    # obj_traces.append(obj_trace[0])

    return var_trace, obj_trace


def CC_Optimization(MAX_iteration, benchmark, scale_range, group, based_population, p, real):
    problem = MyProblem.CC_Problem(group, benchmark, scale_range, based_population)  # 实例化问题对象

    """===========================算法参数设置=========================="""

    myAlgorithm = templet.soea_DE_currentToBest_1_L_templet(problem, p)
    myAlgorithm.MAXGEN = MAX_iteration
    myAlgorithm.drawing = 0
    """=====================调用算法模板进行种群进化====================="""
    # [population, obj_trace, var_trace] = myAlgorithm.run(population, MAX_iteration)
    [population, obj_trace, var_trace] = myAlgorithm.run(real)
    # obj_traces.append(obj_trace[0])

    return var_trace, obj_trace, population
