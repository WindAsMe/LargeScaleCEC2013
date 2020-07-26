import copy
import numpy as np
import math
import geatpy as ea
from in20200507.SparseModelBaseDE.model.MyProblem import MyProblem
from scipy import stats
import random
import matplotlib.pyplot as plt
import time


def find_n_best(num_list, topk=50):
    tmp_list = copy.deepcopy(num_list)
    tmp_list.sort()
    min_num_index = [num_list.index(one) for one in tmp_list[:topk]]
    return min_num_index


def matrix2list(m):
    result = []
    for l in m:
        result.append(l[0])
    return result


def get_chrom(chrom, best_index):
    new_chrom = []
    for i in best_index:
        new_chrom.append(chrom[i])
    return np.array(new_chrom)


def distance(x1, x2, limitation):
    dis = 0
    scale = 0
    for i in range(len(x1)):
        dis += (x1[i] - x2[i]) ** 2
        scale += limitation ** 2
    return math.sqrt(dis) / math.sqrt(scale)


def find_best_worst(aim_two, best_or_worst, Dimension, reg):
    best_indexes = []
    time1 = time.time()
    two_index, two_trace = Optimization(Dimension, aim_two, reg, best_or_worst)
    time2 = time.time()
    print('Sparse model optimization time: ', time2 - time1)
    best_indexes.append(two_index)

    trace = two_trace[:, 1]
    plt.plot(trace, label='sparse model optimization')
    plt.legend()
    plt.show()
    return best_indexes


def Optimization(Dimension, function, reg, max_min):
    problem = MyProblem(Dimension, function, reg, max_min)  # 实例化问题对象

    """==============================种群设置==========================="""
    Encoding = 'RI'  # 编码方式
    NIND = 50  # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
    population = ea.Population(Encoding, Field, NIND)

    """===========================算法参数设置=========================="""
    # population.initChrom(NIND)
    # print(population.Chrom)
    myAlgorithm = ea.soea_DE_best_1_L_templet(problem, population)
    myAlgorithm.MAXGEN = Dimension * 50
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

    return best_index, obj_trace


def create_points(aim, best_index, number):
    best_indexes = [best_index]
    best_indexes_value = [aim(best_index)]

    scale_factors = [0.5, 1, 1.5, 2]
    for scale_factor in scale_factors:
        for i in range(number - 1):
            good_index = copy.deepcopy(best_index)
            for i in range(len(good_index)):
                good_index[i] += random.random() * scale_factor
            if len(best_indexes) < number:
                best_indexes.append(good_index)
                best_indexes_value.append(aim(good_index))
            else:
                worst_index_value = max(best_indexes_value)
                worst_index = best_indexes_value.index(worst_index_value)
                if aim(good_index) <= worst_index_value:
                    best_indexes_value[worst_index] = aim(good_index)
                    best_indexes[worst_index] = good_index
    return best_indexes


def statistic(original, assist_one, assist_near, method='f'):
    s_k, p_k = stats.kruskal(original, assist_one, assist_near)
    s_f, p_f = stats.friedmanchisquare(original, assist_one, assist_near)
    p_k = float("%.2e" % p_k)
    p_f = float("%.2e" % p_f)
    ave_o = sum(original) / len(original)
    ave_one_a = sum(assist_one) / len(assist_one)
    ave_near_a = sum(assist_near) / len(assist_near)
    if method == 'k' or method == 'K':
        return p_k, ave_o, ave_one_a, ave_near_a
    else:
        return p_f, ave_o, ave_one_a, ave_near_a


def write_statistic(path, generation, p_value, original, assist_one, assist_near):
    f = open('D:\CS2019KYUTAI\PythonProject\SparseModeling\in20200507\SparseModelBaseDE\data\\' + path, 'a')
    f.write('In ' + generation + ': ' + 'p-value=' + p_value + '\n')
    f.write(original.__str__() + '\n')
    f.write(assist_one.__str__() + '\n')
    f.write(assist_near.__str__() + '\n')
    f.write('\n')
    f.close()


def write_draw(path, original, assist_one, assist_near):
    f = open('D:\CS2019KYUTAI\PythonProject\SparseModeling\in20200507\SparseModelBaseDE\data\\' + path, 'a')
    f.write(original.__str__() + '\n')
    f.write(assist_one.__str__() + '\n')
    f.write(assist_near.__str__() + '\n')
    f.write('\n')
    f.close()


def write_evaluate(path, name, d):
    f = open('D:\CS2019KYUTAI\PythonProject\SparseModeling\in20200507\SparseModelBaseDE\data\\' + path, 'a')
    f.write(name + '\n')
    f.write('\n')
    for i in d:
        flag = i.split('_')
        f.write('Population Size: ' + flag[0] + ' Max Iteration: ' + flag[1] + '\n')
        f.write(d[i].__str__() + '\n')
        f.write('Average: ' + str(sum(d[i]) / len(d[i])) + '\n')
    f.close()


def write_raw_data(path, round, original, assist_one, assist_near):
    f = open('D:\CS2019KYUTAI\PythonProject\SparseModeling\in20200507\SparseModelBaseDE\data\\' + path, 'a')
    f.write('Round ' + str(round) + '\n')

    f.write(str(original) + '\n')
    f.write(str(assist_one) + '\n')
    f.write(str(assist_near) + '\n')
    f.write('\n')
    f.close()



