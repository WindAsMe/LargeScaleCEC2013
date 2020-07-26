from in20200530.SparseModelContourf.model.MyProblem import MyProblem
from in20200530.SparseModelContourf.util import help
import geatpy as ea
import numpy as np


def OptimizationForSparse(Dimension, function, reg, max_min):
    problem = MyProblem(Dimension, function, reg, max_min)  # 实例化问题对象

    """==============================种群设置==========================="""
    Encoding = 'RI'  # 编码方式
    NIND = 10  # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
    population = ea.Population(Encoding, Field, NIND)

    """===========================算法参数设置=========================="""
    # population.initChrom(NIND)
    # print(population.Chrom)
    myAlgorithm = ea.soea_DE_best_1_L_templet(problem, population)
    myAlgorithm.MAXGEN = 100
    myAlgorithm.mutOper.F = 0.5
    myAlgorithm.recOper.XOVR = 0.5
    myAlgorithm.drawing = 0
    """=====================调用算法模板进行种群进化====================="""
    [population, obj_trace, var_trace] = myAlgorithm.run()
    best_gen = np.argmax(obj_trace[:, 1])
    best_ObjV = obj_trace[best_gen, 1]
    # print('最优的目标函数值为：%s' % (best_ObjV))
    # print('最优的决策变量值为：')
    best_index = []
    for i in range(var_trace.shape[1]):
        best_index.append(var_trace[best_gen, i])
    # print(best_index)
    # print('有效进化代数：%s' % (obj_trace.shape[0]))
    # print('最优的一代是第 %s 代' % (best_gen + 1))
    # print('评价次数：%s' % (myAlgorithm.evalsNum))
    # print('时间已过 %s 秒' % (myAlgorithm.passTime))

    return best_index

