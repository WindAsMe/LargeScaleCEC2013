from in20200621.DimensionRedunctionForSparse.DE import MyProblem
import geatpy as ea
import numpy as np
import copy


def SimpleProblemsOptimization(Dim, NIND, MAX_iteration, benchmark_function, init_population, function, groups, feature_names, coef, max_min):
    var_traces = np.zeros((MAX_iteration, Dim))

    for i in range(len(groups)):
        var_trace = help_SimpleProblemsOptimization(Dim, NIND, MAX_iteration, init_population, function,
                                                    benchmark_function, groups[i], max_min)

        for element in groups[i]:
            var_traces[:, element] = var_trace[:, element]
    var_traces = np.array(var_traces, dtype='float16')

    obj_traces = []
    for var_trace in var_traces:
        obj_traces.append(benchmark_function(var_trace))

    return obj_traces, var_traces[len(var_traces)-1, :]


def help_SimpleProblemsOptimization(Dimension, NIND, MAX_iteration, init_population, function, benchmark, group, max_min):
    problem = MyProblem.MySimpleProblem(Dimension, function, group, benchmark, max_min)  # 实例化问题对象

    """==============================种群设置==========================="""
    Encoding = 'RI'  # 编码方式
    NIND = NIND * len(group)  # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
    population = ea.Population(Encoding, Field, NIND)

    """===========================算法参数设置=========================="""
    population.initChrom(NIND)
    population.Chrom = copy.deepcopy(init_population)

    myAlgorithm = ea.soea_DE_best_1_L_templet(problem, population)
    myAlgorithm.MAXGEN = MAX_iteration
    myAlgorithm.mutOper.F = 0.5
    myAlgorithm.recOper.XOVR = 0.5
    myAlgorithm.drawing = 0
    """=====================调用算法模板进行种群进化====================="""
    [population, obj_trace, var_trace] = myAlgorithm.run()
    return var_trace


def ComplexProblemsOptimization(Dimension, NIND, MAX_iteration, init_population, function, benchmark, max_min):
    problem = MyProblem.MyComplexProblem(Dimension, function, benchmark, max_min)  # 实例化问题对象

    """==============================种群设置==========================="""
    Encoding = 'RI'  # 编码方式
    NIND = NIND  # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
    population = ea.Population(Encoding, Field, NIND)

    """===========================算法参数设置=========================="""
    population.initChrom(NIND)
    population.Chrom = copy.deepcopy(init_population)

    myAlgorithm = ea.soea_DE_best_1_L_templet(problem, population)
    myAlgorithm.MAXGEN = MAX_iteration
    myAlgorithm.mutOper.F = 0.5
    myAlgorithm.recOper.XOVR = 0.5
    myAlgorithm.drawing = 0
    """=====================调用算法模板进行种群进化====================="""
    [population, obj_trace, var_trace] = myAlgorithm.run()
    return obj_trace[:, 1]
