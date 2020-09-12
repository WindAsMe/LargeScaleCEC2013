from in20200911.DimensionReductionForSparse.DE import MyProblem
from in20200911.DimensionReductionForSparse.util import help
import geatpy as ea
import numpy as np


def ProblemsOptimization(Dim, NIND, MAX_iteration, benchmark, scale_range, groups, initial_population):
    var_traces = np.zeros((MAX_iteration, Dim))
    based_population = np.zeros(Dim)
    for i in range(len(groups)):
        certain_initial_population = np.vstack(initial_population[groups[i][0] * NIND:(groups[i][0] + 1) * NIND, :])
        if len(groups[i]) > 1:
            for j in range(1, len(groups[i])):
                certain_initial_population = np.vstack((certain_initial_population,
                                                        initial_population[groups[i][j] * NIND:(groups[i][j] + 1) * NIND, :]))

        var_trace, obj_trace = GroupOptimization(Dim, NIND, MAX_iteration, benchmark, scale_range, groups[i],
                                                 based_population, certain_initial_population)
        # print(var_trace.shape)
        # print('    Finished: ', i + 1, '/', len(groups))
        # print(np.argmin(obj_trace[:, 1]))
        for element in groups[i]:
            var_traces[:, element] = var_trace[:, element]
            based_population[element] = var_trace[np.argmin(obj_trace[:, 1]), element]

        x = np.linspace(0, len(groups[i]) * MAX_iteration * NIND, MAX_iteration)
        help.draw_obj(x, obj_trace[:, 1], 'label', 'temp')
    var_traces, obj_traces = help.preserve(var_traces, benchmark)
    return obj_traces, var_traces


# Optimization for one group
def GroupOptimization(Dim, NIND, MAX_iteration, benchmark, scale_range, group, based_population, certain_initial_population):
    problem = MyProblem.MySimpleProblem(Dim, group, benchmark, scale_range, NIND * len(group), based_population)  # 实例化问题对象

    """==============================种群设置==========================="""
    Encoding = 'RI'  # 编码方式
    NIND = NIND * len(group)  # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
    population = ea.Population(Encoding, Field, NIND)
    population.initChrom(NIND)
    population.Chrom = certain_initial_population
    population.Phen = certain_initial_population
    help.set_Chrom_zero(population, group, benchmark)

    print('verify: ', population.Chrom)
    """===========================算法参数设置=========================="""

    # myAlgorithm = templet.soea_SaNSDE_templet(problem, population)
    myAlgorithm = ea.soea_DE_currentToBest_1_L_templet(problem, population)
    myAlgorithm.MAXGEN = MAX_iteration
    myAlgorithm.drawing = 0
    """=====================调用算法模板进行种群进化====================="""
    # [population, obj_trace, var_trace] = myAlgorithm.run(population, MAX_iteration)
    [population, obj_trace, var_trace] = myAlgorithm.run(population)
    print(obj_trace)
    # obj_traces.append(obj_trace[0])
    return var_trace, obj_trace

