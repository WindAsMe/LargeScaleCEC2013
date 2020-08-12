from in20200808.DimensionReductionForSparse.DE import MyProblem
from in20200808.DimensionReductionForSparse.parameters import f_num
import geatpy as ea
import numpy as np


def SimpleProblemsOptimization(Dim, NIND, MAX_iteration, benchmark_function, scale_range, function, groups, max_min):
    var_traces = np.zeros((MAX_iteration, Dim))
    iteration_times = 0
    for i in range(len(groups)):
        var_trace = help_SimpleProblemsOptimization(Dim, NIND, MAX_iteration, function, benchmark_function, scale_range,
                                                    groups[i], max_min)
        print('    Finished: ', i+1, '/', len(groups))
        iteration_times += len(var_trace) * NIND * len(groups[i])

        for element in groups[i]:
            var_traces[:, element] = var_trace[:, element]

    var_traces = np.array(var_traces, dtype='float16')

    obj_traces = []
    for var_trace in var_traces:
        obj_traces.append(benchmark_function(var_trace, f_num))

    return obj_traces, var_traces[len(var_traces)-1, :], iteration_times


def help_SimpleProblemsOptimization(Dimension, NIND, MAX_iteration, function, benchmark, scale_range, group, max_min):
    problem = MyProblem.MySimpleProblem(Dimension, function, group, benchmark, scale_range, max_min)  # 实例化问题对象

    """==============================种群设置==========================="""
    Encoding = 'RI'  # 编码方式
    NIND = NIND * len(group)  # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
    population = ea.Population(Encoding, Field, NIND)

    """===========================算法参数设置=========================="""
    population.initChrom(NIND)

    myAlgorithm = ea.soea_DE_rand_1_L_templet(problem, population)
    myAlgorithm.MAXGEN = MAX_iteration
    myAlgorithm.mutOper.F = 0.5
    myAlgorithm.recOper.XOVR = 0.5
    myAlgorithm.drawing = 0
    """=====================调用算法模板进行种群进化====================="""
    [population, obj_trace, var_trace] = myAlgorithm.run()
    return var_trace


def ComplexProblemsOptimization(Dimension, NIND, MAX_iteration, function, benchmark, scale_range, max_min):
    problem = MyProblem.MyComplexProblem(Dimension, function, benchmark, scale_range, max_min)  # 实例化问题对象

    """==============================种群设置==========================="""
    Encoding = 'RI'  # 编码方式
    NIND = NIND  # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
    population = ea.Population(Encoding, Field, NIND)

    """===========================算法参数设置=========================="""
    population.initChrom(NIND)

    myAlgorithm = ea.soea_DE_rand_1_L_templet(problem, population)
    myAlgorithm.MAXGEN = MAX_iteration
    myAlgorithm.mutOper.F = 0.5
    myAlgorithm.recOper.XOVR = 0.5
    myAlgorithm.drawing = 0
    """=====================调用算法模板进行种群进化====================="""
    [population, obj_trace, var_trace] = myAlgorithm.run()

    return obj_trace, len(obj_trace[:, 1]) * NIND
