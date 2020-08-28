from in20200828.DimensionReductionForSparse.DE import MyProblem
from in20200828.DimensionReductionForSparse.util import help
import geatpy as ea
import numpy as np


def SimpleProblemsOptimization(Dim, NIND, MAX_iteration, benchmark, scale_range, groups, max_min):
    var_traces = np.zeros((MAX_iteration, Dim))
    based_population = np.zeros(Dim)
    for i in range(len(groups)):
        var_trace = help_SimpleProblemsOptimization(Dim, NIND, MAX_iteration, benchmark, scale_range,
                                                    groups[i], max_min, based_population)
        # print(var_trace.shape)
        print('    Finished: ', i + 1, '/', len(groups))
        for element in groups[i]:
            var_traces[:, element] = var_trace[:, element]
            based_population[element] = var_trace[len(var_trace) - 1, element]

        # d = []
        # for v in var_traces:
        #     d.append(benchmark(v))
        # x = np.linspace(0, 3000000, 30, endpoint=False)
        # help.draw_obj(x, d, 'temp')
    # var_traces = help.preserve(var_traces, benchmark)
    obj_traces = []
    for var_trace in var_traces:
        obj_traces.append(benchmark(var_trace))

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
    p = 0.5
    fp = 0.5
    var_traces = []

    for i in range(MAX_iteration):
        # if help.DE_choice(p):
        #     myAlgorithm = ea.soea_DE_rand_1_L_templet(problem, population)
        # else:
        myAlgorithm = ea.soea_DE_currentToBest_1_L_templet(problem, population)

        myAlgorithm.MAXGEN = 1
        myAlgorithm.mutOper.F = fp
        myAlgorithm.recOper.XOVR = 0.5
        myAlgorithm.drawing = 0
        """=====================调用算法模板进行种群进化====================="""
        [population, obj_trace, var_trace] = myAlgorithm.run(population)
        var_traces.append(var_trace[0])

    return np.array(var_traces)


