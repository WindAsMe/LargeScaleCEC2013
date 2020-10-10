from in20200913.DimensionReductionForSparse.DE import MyProblem, templet
from in20200913.DimensionReductionForSparse.util import help
import geatpy as ea
import numpy as np


def ProblemsOptimization(Dim, NIND, MAX_iteration, benchmark, scale_range, groups):
    var_traces = np.zeros((MAX_iteration, Dim))
    based_population = np.zeros(Dim)
    for i in range(len(groups)):
        var_trace, obj_trace = GroupOptimization(NIND, MAX_iteration, benchmark, scale_range, groups[i],
                                                 based_population)

        for element in groups[i]:
            var_traces[:, element] = var_trace[:, groups[i].index(element)]
            based_population[element] = var_trace[np.argmin(obj_trace[:, 1]), groups[i].index(element)]

        # x = np.linspace(0, len(groups[i]) * MAX_iteration * NIND, MAX_iteration)
        # help.draw_obj(x, obj_trace[:, 1], 'method', 'temp')
    var_traces, obj_traces = help.preserve(var_traces, benchmark)
    return var_traces, obj_traces


# Optimization for one group
def GroupOptimization(NIND, MAX_iteration, benchmark, scale_range, group, based_population):
    problem = MyProblem.MySimpleProblem(group, benchmark, scale_range, NIND * len(group), based_population)  # 实例化问题对象

    """==============================种群设置==========================="""
    Encoding = 'RI'  # 编码方式
    NIND = NIND * len(group)  # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
    population = ea.Population(Encoding, Field, NIND)
    population.initChrom(NIND)

    """===========================算法参数设置=========================="""

    # myAlgorithm = templet.soea_SaNSDE_templet(problem, population)
    myAlgorithm = templet.soea_currentToBest_templet(problem, population)
    myAlgorithm.MAXGEN = MAX_iteration
    myAlgorithm.drawing = 0
    """=====================调用算法模板进行种群进化====================="""
    # [population, obj_trace, var_trace] = myAlgorithm.run(population, MAX_iteration)
    [population, obj_trace, var_trace] = myAlgorithm.run(population)
    # obj_traces.append(obj_trace[0])
    return var_trace, obj_trace


