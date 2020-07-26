import numpy as np
import geatpy as ea
from before20200507.SparseModelingSort import SparseTest, benchmark, aims

import math


class MyProblem(ea.Problem):
    def __init__(self, Dim, function, reg, max_min):
        name = 'MyProblem'
        M = 1
        maxormins = [max_min]
        varTypes = [0] * Dim
        lb = [-10] * Dim
        ub = [10] * Dim
        lbin = [1] * Dim
        ubin = [1] * Dim
        self.function = function
        self.reg = reg
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):                 # 目标函数，pop为传入的种群对象
        pop.ObjV = self.function(pop.Phen, self.reg)


def Optimization(Dimension, function, reg, max_min):
    problem = MyProblem(Dimension, function, reg, max_min)  # 实例化问题对象

    """==============================种群设置==========================="""
    Encoding = 'RI'  # 编码方式
    NIND = 50  # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
    population = ea.Population(Encoding, Field, NIND)


    """===========================算法参数设置=========================="""
    population.initChrom(NIND)
    print(population.Chrom)
    myAlgorithm = ea.soea_DE_best_1_L_templet(problem, population)
    myAlgorithm.MAXGEN = 5000
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


def distance(x1, x2, limitation):
    dis = 0
    scale = 0
    for i in range(len(x1)):
        dis += (x1[i] - x2[i]) ** 2
        scale += limitation ** 2
    return math.sqrt(dis) / math.sqrt(scale)


def find_best_worst(aim_original, aim_two, iteration, best_or_worst, reg, Dimension):
    average = []
    min_index = []
    min_distance = 1
    for i in range(iteration):
        # print(i)
        original_index = Optimization(Dimension, aim_original, reg, best_or_worst)
        two_index = Optimization(Dimension, aim_two, reg, for_min)

        dis = distance(original_index, two_index, 20)
        average.append(distance(original_index, two_index, 20))
        if dis < min_distance:
            min_distance = dis
            min_index = two_index

    if best_or_worst == 1:
        print('min Average: ', sum(average) / iteration, 'Max: ', max(average), 'Min: ', min(average))
    if best_or_worst == -1:
        print('max Average: ', sum(average) / iteration, 'Max: ', max(average), 'Min: ', min(average))

    return min_index


if __name__ == '__main__':
    Dimension = 10
    for_max = -1
    for_min = 1
    scale = 20
    iteration = 50

    aim_original = aims.aim_Schwefel_original
    aim_two = aims.aim_Schwefel_two
    benchmark_function = benchmark.Schwefel

    # Schwefel: min
    # Rosenbrock: min
    # Rastrigin: min
    # Ackley: min
    reg = SparseTest.SparseModeling(Dimension, benchmark_function, 2)

    # As prior knowledge
    best_index = find_best_worst(aim_original, aim_two, iteration, for_min, reg, Dimension)
    # find_best_worst(iteration, for_max, reg, Dimension)


