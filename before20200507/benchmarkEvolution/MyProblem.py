import numpy as np
import geatpy as ea


def aim_Griewank_original(Phen):
    part1 = 0
    part2 = 1
    for i in range(0, len(Phen[0])):
        part1 += Phen[:, [i]] ** 2 / 4000
        part2 *= np.cos(Phen[:, [i]] / np.sqrt(i + 1))
    return part1 - part2 + 1


class MyProblem(ea.Problem):
    def __init__(self, Dim):
        name = 'MyProblem'
        M = 1
        maxormins = [-1]
        varTypes = [0] * Dim
        lb = [-10] * Dim
        ub = [10] * Dim
        lbin = [1] * Dim
        ubin = [1] * Dim
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):                 # 目标函数，pop为传入的种群对象
        pop.ObjV = aim_Griewank_original(pop.Phen)


def Optimization(Dimension):
    problem = MyProblem(Dimension)  # 实例化问题对象

    """==============================种群设置==========================="""
    Encoding = 'RI'  # 编码方式
    NIND = 50  # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
    population = ea.Population(Encoding, Field, NIND)

    """===========================算法参数设置=========================="""
    myAlgorithm = ea.soea_DE_best_1_L_templet(problem, population)
    myAlgorithm.MAXGEN = 1000
    myAlgorithm.mutOper.F = 0.5
    myAlgorithm.recOper.XOVR = 0.5
    myAlgorithm.drawing = 1

    """=====================调用算法模板进行种群进化====================="""
    [population, obj_trace, var_trace] = myAlgorithm.run()
    best_gen = np.argmax(obj_trace[:, 1])
    best_ObjV = obj_trace[best_gen, 1]
    print('最优的目标函数值为：%s' % (best_ObjV))
    print('最优的决策变量值为：')
    for i in range(var_trace.shape[1]):
        print(var_trace[best_gen, i])
    print('有效进化代数：%s' % (obj_trace.shape[0]))
    print('最优的一代是第 %s 代' % (best_gen + 1))
    print('评价次数：%s' % (myAlgorithm.evalsNum))
    print('时间已过 %s 秒' % (myAlgorithm.passTime))


if __name__ == '__main__':
    Optimization(10)
