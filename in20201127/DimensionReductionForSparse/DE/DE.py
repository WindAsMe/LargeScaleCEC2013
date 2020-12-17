from in20201127.DimensionReductionForSparse.DE import MyProblem
from in20201127.DimensionReductionForSparse.util import help
import geatpy as ea
import numpy as np


def ProblemsOptimization(Dim, NIND, MAX_iteration, benchmark, scale_range, groups, maxormin):
    var_traces = np.zeros((MAX_iteration, Dim))
    based_population = np.zeros(Dim)
    for i in range(len(groups)):
        var_trace, obj_trace = GroupOptimization(NIND, MAX_iteration, benchmark, scale_range, groups[i],
                                                 based_population, maxormin)

        for element in groups[i]:
            var_traces[:, element] = var_trace[:, groups[i].index(element)]
            based_population[element] = var_trace[np.argmin(obj_trace[:, 1]), groups[i].index(element)]

        # x = np.linspace(0, len(groups[i]) * MAX_iteration * NIND, MAX_iteration)
        # help.draw_obj(x, obj_trace[:, 1], 'method', 'temp')
    var_traces, obj_traces = help.preserve(var_traces, benchmark)
    return var_traces, obj_traces


# Optimization for one group
def GroupOptimization(NIND, MAX_iteration, benchmark, scale_range, group, based_population, maxormin):
    problem = MyProblem.MySimpleProblem(group, benchmark, scale_range, NIND * len(group), maxormin, based_population)  # 实例化问题对象

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
    return var_trace, obj_trace


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