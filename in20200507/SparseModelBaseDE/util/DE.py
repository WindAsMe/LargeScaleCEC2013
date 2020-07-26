from in20200507.SparseModelBaseDE.model.MyProblem import MyProblem
from in20200507.SparseModelBaseDE.util import help
import geatpy as ea
import numpy as np


def NoAssistOptimization(Dimension, function, reg, max_min):
    problem = MyProblem(Dimension, function, reg, max_min)  # 实例化问题对象

    """==============================种群设置==========================="""
    Encoding = 'RI'  # 编码方式
    NIND = 100  # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
    population = ea.Population(Encoding, Field, NIND)

    """===========================算法参数设置=========================="""
    population.initChrom(NIND)
    # print(population.Chrom)
    init_Chrom = population.Chrom
    myAlgorithm = ea.soea_DE_best_1_L_templet(problem, population)
    myAlgorithm.MAXGEN = Dimension * 100 + 200
    myAlgorithm.mutOper.F = 0.5
    myAlgorithm.recOper.XOVR = 0.5
    myAlgorithm.drawing = 0
    """=====================调用算法模板进行种群进化====================="""
    [population, obj_trace, var_trace] = myAlgorithm.run()
    best_gen = np.argmin(obj_trace[:, 1])
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
    return best_index, obj_trace, init_Chrom


def AssistOptimization(Dimension, function, reg, max_min, best_indexes, init_Chrom):
    problem = MyProblem(Dimension, function, reg, max_min)  # 实例化问题对象

    """==============================种群设置==========================="""
    Encoding = 'RI'  # 编码方式
    NIND = 100  # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
    population = ea.Population(Encoding, Field, NIND)

    """===========================算法参数设置=========================="""
    # Assist evolution
    # Control the initial chrom is same
    population.initChrom(NIND)
    population.Chrom = init_Chrom
    matrix_combination = np.append(population.Chrom, best_indexes, axis=0)
    first_evaluate = function(matrix_combination, reg)
    first_evaluate_list = help.matrix2list(first_evaluate)
    chrom_index = help.find_n_best(first_evaluate_list, NIND)

    chrom = help.get_chrom(matrix_combination, chrom_index)
    print('Assist: best in random', sorted(function(init_Chrom, reg))[0])
    print('Assist: the prior knowledge nearby ', sorted(function(np.array(best_indexes), reg)))

    population.Chrom = chrom

    myAlgorithm = ea.soea_DE_best_1_L_templet(problem, population)
    myAlgorithm.MAXGEN = Dimension * 100 + 200
    myAlgorithm.mutOper.F = 0.5
    myAlgorithm.recOper.XOVR = 0.5
    myAlgorithm.drawing = 0
    """=====================调用算法模板进行种群进化====================="""
    [population, obj_trace, var_trace] = myAlgorithm.run()
    best_gen = np.argmin(obj_trace[:, 1])
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
    return best_index, obj_trace
