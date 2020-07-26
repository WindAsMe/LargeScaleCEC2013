from in20200507.SparseModelBaseDE.util.SparseTest import SparseModeling
from in20200507.SparseModelBaseDE.model.MyProblem import MyProblem
from in20200507.SparseModelBaseDE.util import benchmark, aims, help
import geatpy as ea
import numpy as np


def Optimization(N, MAX, Dimension, function, reg, max_min):
    problem = MyProblem(Dimension, function, reg, max_min)  # 实例化问题对象

    """==============================种群设置==========================="""
    Encoding = 'RI'  # 编码方式
    NIND = N  # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
    population = ea.Population(Encoding, Field, NIND)

    """===========================算法参数设置=========================="""
    # population.initChrom(NIND)
    # print(population.Chrom)
    myAlgorithm = ea.soea_DE_best_1_L_templet(problem, population)
    myAlgorithm.MAXGEN = MAX
    myAlgorithm.mutOper.F = 0.5
    myAlgorithm.recOper.XOVR = 0.5
    myAlgorithm.drawing = 0
    """=====================调用算法模板进行种群进化====================="""
    [population, obj_trace, var_trace] = myAlgorithm.run()
    best_gen = np.argmax(obj_trace[:, 1])
    best_ObjV = obj_trace[best_gen, 1]

    return float("%.3e" % best_ObjV)


if __name__ == '__main__':
    Dimension = 100
    benchmark_function = benchmark.Ackley
    aim = aims.aim_Ackley_two
    reg = SparseModeling(Dimension, benchmark_function, 2)
    test_times = 10
    name = 'At '+str(Dimension)+'D: Ackley'

    N = [10, 20, 50, 100]
    MAX = [100, 500, 1000, 2000, 5000, 10000]

    d = {}
    for i in range(test_times):
        print('round ', i + 1)
        for n in N:
            for m in MAX:
                result = Optimization(n, m, Dimension, aim, reg, 1)
                if str(n)+'_'+str(m) in d:
                    l = d[str(n) + '_' + str(m)]
                    l.append(result)
                    d[str(n)+'_'+str(m)] = l
                else:
                    d[str(n) + '_' + str(m)] = [result]

    help.write_evaluate('sparse_optimization', name, d)



