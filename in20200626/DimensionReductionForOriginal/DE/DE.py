from in20200626.DimensionReductionForOriginal.DE import MyProblem
import geatpy as ea
import numpy as np


def SimpleProblemsOptimization(Dim, NIND, MAX_iteration,  benchmark_function, evaluate_function, groups, intercept, max_min):
    index = []
    for i in range(MAX_iteration):
        index.append([0] * Dim)
    index = np.array(index, dtype="float64")
    trace_combination = []
    trace_all = []
    for i in range(len(groups)):
        obj_trace, var_trace = help_SimpleProblemsOptimization(Dim, NIND * len(groups[i]), MAX_iteration,
                                                                 benchmark_function, evaluate_function, groups[i], intercept, max_min)

        for element in groups[i]:
            index[:, element] = var_trace[:, element]
        trace_all.append(obj_trace)

    trace_all = np.array(trace_all)
    for i in range(len(trace_all[0])):
        trace_combination.append(sum(trace_all[:, i]) + intercept)
    trace_combination = np.array(trace_combination)
    return trace_combination, index


def help_SimpleProblemsOptimization(Dimension, NIND, MAX_iteration, benchmark_function, evaluate_function, group, intercept, max_min):
    problem = MyProblem.MySimpleProblem(Dimension, benchmark_function, evaluate_function, group, intercept, max_min)  # 实例化问题对象

    """==============================种群设置==========================="""
    Encoding = 'RI'  # 编码方式
    NIND = NIND  # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
    population = ea.Population(Encoding, Field, NIND)

    """===========================算法参数设置=========================="""

    myAlgorithm = ea.soea_DE_best_1_L_templet(problem, population)
    myAlgorithm.MAXGEN = MAX_iteration
    myAlgorithm.mutOper.F = 0.5
    myAlgorithm.recOper.XOVR = 0.5
    myAlgorithm.drawing = 0
    """=====================调用算法模板进行种群进化====================="""
    [population, obj_trace, var_trace] = myAlgorithm.run()
    best_gen = np.argmin(obj_trace[:, 1])
    best_ObjV = obj_trace[best_gen, 1]
    # print('最优的目标函数值为：%s' % (best_ObjV))
    # print('最优的决策变量值为：')
    # best_index = []
    # for i in range(var_trace.shape[1]):
    #     best_index.append(var_trace[best_gen, i])
    # print(best_index)
    # print('有效进化代数：%s' % (obj_trace.shape[0]))
    # print('最优的一代是第 %s 代' % (best_gen + 1))
    # print('评价次数：%s' % (myAlgorithm.evalsNum))
    # print('时间已过 %s 秒' % (myAlgorithm.passTime))
    return obj_trace[:, 1], np.array(var_trace)


def ComplexProblemsOptimization(Dimension, NIND, MAX_iteration, benchmark_function, evaluate_function, max_min):
    problem = MyProblem.MyComplexProblem(Dimension, benchmark_function, evaluate_function, max_min)  # 实例化问题对象

    """==============================种群设置==========================="""
    Encoding = 'RI'  # 编码方式
    NIND = NIND  # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
    population = ea.Population(Encoding, Field, NIND)

    """===========================算法参数设置=========================="""

    myAlgorithm = ea.soea_DE_best_1_L_templet(problem, population)
    myAlgorithm.MAXGEN = MAX_iteration
    myAlgorithm.mutOper.F = 0.5
    myAlgorithm.recOper.XOVR = 0.5
    myAlgorithm.drawing = 0
    """=====================调用算法模板进行种群进化====================="""
    [population, obj_trace, var_trace] = myAlgorithm.run()
    best_gen = np.argmin(obj_trace[:, 1])
    best_ObjV = obj_trace[best_gen, 1]
    # print('最优的目标函数值为：%s' % (best_ObjV))
    # print('最优的决策变量值为：')
    # best_index = []
    # for i in range(var_trace.shape[1]):
    #     best_index.append(var_trace[best_gen, i])
    # print(best_index)
    # print('有效进化代数：%s' % (obj_trace.shape[0]))
    # print('最优的一代是第 %s 代' % (best_gen + 1))
    # print('评价次数：%s' % (myAlgorithm.evalsNum))
    # print('时间已过 %s 秒' % (myAlgorithm.passTime))
    return obj_trace[:, 1], var_trace
