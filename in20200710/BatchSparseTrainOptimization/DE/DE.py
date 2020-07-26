from in20200710.BatchSparseTrainOptimization.DE import MyProblem
import geatpy as ea
import numpy as np
import time


def SimpleProblemsOptimization(func_num, Dim, NIND, MAX_iteration, benchmark_function, evaluate_function, groups, max_min):
    var_traces = []
    obj_traces = []
    for i in range(MAX_iteration):
        var_traces.append([0] * Dim)
    var_traces = np.array(var_traces, dtype="float16")
    total_task = len(groups)
    current_task = 1
    for group in groups:
        time1 = time.process_time()
        var_trace = help_SimpleProblemsOptimization(func_num, Dim, NIND * len(group), MAX_iteration, benchmark_function,
                                                    evaluate_function, group, max_min)
        time2 = time.process_time()

        print('finished: ', current_task, '/', total_task, ' Time consuming: ', time2 - time1)
        current_task += 1
        for element in group:
            var_traces[:, element] = var_trace[:, element]
    for var in var_traces:
        obj_traces.append(benchmark_function(var, func_num))

    return obj_traces, var_traces[len(var_traces) - 1]


def help_SimpleProblemsOptimization(func_num, Dim, NIND, MAX_iteration, benchmark_function, evaluate_function, group, max_min):
    problem = MyProblem.MySimpleProblem(func_num, Dim, benchmark_function, evaluate_function, group, max_min)  # 实例化问题对象

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
    return var_trace


def ComplexProblemsOptimization(Dim, NIND, MAX_iteration, benchmark_function, evaluate_function, max_min):
    problem = MyProblem.MyComplexProblem(Dim, benchmark_function, evaluate_function, max_min)  # 实例化问题对象

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
    return obj_trace[:, 1]
