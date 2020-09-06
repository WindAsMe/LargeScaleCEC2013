from in20200828.DimensionReductionForSparse.DE import MyProblem, templet
from in20200828.DimensionReductionForSparse.util import help
import geatpy as ea
import numpy as np


def SimpleProblemsOptimization(Dim, NIND, MAX_iteration, benchmark, scale_range, groups, max_min):
    var_traces = np.zeros((MAX_iteration, Dim))
    based_population = np.zeros(Dim)
    for i in range(len(groups)):
        var_trace, obj_trace = help_SimpleProblemsOptimization(Dim, NIND, MAX_iteration, benchmark, scale_range,
                                                    groups[i], max_min, based_population)
        # print(var_trace.shape)
        print('    Finished: ', i + 1, '/', len(groups))
        # print(np.argmin(obj_trace[:, 1]))
        for element in groups[i]:
            var_traces[:, element] = var_trace[:, element]
            based_population[element] = var_trace[np.argmin(obj_trace[:, 1]), element]

        # x = np.linspace(0, len(groups[i]) * MAX_iteration * NIND, MAX_iteration)
        # help.draw_obj(x, obj_trace[:, 1], 'temp')
    var_traces, obj_traces = help.preserve(var_traces, benchmark)
    return obj_traces, var_traces


# Optimization for one group
def help_SimpleProblemsOptimization(Dimension, NIND, MAX_iteration, benchmark, scale_range, group, max_min, based_population):
    problem = MyProblem.MySimpleProblem(Dimension, group, benchmark, scale_range, max_min, NIND * len(group), based_population)  # 实例化问题对象

    """==============================种群设置==========================="""
    Encoding = 'RI'  # 编码方式
    NIND = NIND * len(group)  # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
    population = ea.Population(Encoding, Field, NIND)
    population.initChrom(NIND)
    help.set_Chrom_zero(population, group, benchmark)

    """===========================算法参数设置=========================="""

    # myAlgorithm = templet.soea_SaNSDE_templet(problem, population)
    myAlgorithm = ea.soea_DE_currentToBest_1_L_templet(problem, population)
    myAlgorithm.MAXGEN = MAX_iteration
    myAlgorithm.drawing = 0
    """=====================调用算法模板进行种群进化====================="""
    # [population, obj_trace, var_trace] = myAlgorithm.run(population, MAX_iteration)
    [population, obj_trace, var_trace] = myAlgorithm.run(population)
    # obj_traces.append(obj_trace[0])
    return var_trace, obj_trace
