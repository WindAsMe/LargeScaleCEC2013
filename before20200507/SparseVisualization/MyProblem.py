import numpy as np
import geatpy as ea
from before20200507.SparseVisualization import temp_benchmark, aims
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import random
import math
from before20200507.SparseModelingSort import benchmark


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
    myAlgorithm = ea.soea_DE_best_1_L_templet(problem, population)
    myAlgorithm.MAXGEN = 10000
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
    return best_index, best_ObjV


def distance(x1, x2, limitation):
    dis = 0
    scale = 0
    for i in range(len(x1)):
        dis += (x1[i] - x2[i]) ** 2
        scale += limitation ** 2
    return math.sqrt(dis) / math.sqrt(scale)


def create_data(scale, dim):
    data = []
    for j in range(0, scale):
        temp = []
        for i in range(0, dim):
            temp.append(random.uniform(-10, 10))
        data.append(temp)

    return data


def create_result(train_data, f):
    result = []
    for x in train_data:
        result.append(f(x))
    return result


def Regression(function):
    train_data = create_data(500, 2)
    train_result = create_result(train_data, function)
    poly_reg = PolynomialFeatures(degree=2)
    train_data_poly = poly_reg.fit_transform(train_data)

    # Tag the vars name with combination
    # feature_name = poly_reg.get_feature_names(input_features=get_feature_name(feature_size))
    feature_name = poly_reg.get_feature_names(input_features=['x1', 'x2'])

    reg = linear_model.Lasso()
    reg.fit(train_data_poly, train_result)
    print(reg.coef_, reg.intercept_)
    print(feature_name)
    return reg


def predict(reg, test_data):
    poly_reg = PolynomialFeatures(degree=2)
    test_data_poly = poly_reg.fit_transform(test_data)
    return reg.predict(test_data_poly)


def matrix_reshape(x1_list, x2_list):
    result = []
    for x1 in x1_list:
        for x2 in x2_list:
            result.append([x1, x2])
    return result


def draw(original_index, original_individual, regression_index, regression_individual, x, y, z1, z2):
    mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    ax.plot_surface(x, y, z1, cmap=plt.cm.coolwarm)
    ax.plot_surface(x, y, z2, cmap=plt.cm.summer)

    ax.plot([original_index[0], regression_index[0]], [original_index[1], regression_index[1]],
            [original_individual, regression_individual], '.', color='black', zorder=10)

    plt.show()


def draw_one(original_index, original_individual, x, y, z1):
    mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.plot_surface(x, y, z1, cmap=plt.cm.summer, label='Regression Function')

    ax.plot([original_index[0]], [original_index[1]],
            [original_individual], '.', color='black', zorder=10)

    plt.show()


if __name__ == '__main__':
    Dimension = 2
    for_max = -1
    for_min = 1
    reg = Regression(benchmark.Ackley)

    original_max_index, original_best = Optimization(Dimension, aims.aim_Ackley_original, reg, for_max)
    original_min_index, original_worst = Optimization(Dimension, aims.aim_Ackley_original, reg, for_min)
    two_max_index, two_best = Optimization(Dimension, aims.aim_Ackley_two, reg, for_max)
    two_min_index, two_worst = Optimization(Dimension, aims.aim_Ackley_two, reg, for_min)

    x = np.linspace(-10, 10, 5000)
    y = np.linspace(-10, 10, 5000)

    temp_xy = matrix_reshape(x, y)
    x, y = np.meshgrid(x, y)

    z1 = temp_benchmark.Ackley(x, y)

    z2 = predict(reg, temp_xy)
    z2 = np.array(z2).reshape(5000, 5000)

    print('for best:')
    print(original_max_index, original_best)
    print(two_max_index, two_best)
    print('for worst:')
    print(original_min_index, original_worst)
    print(two_min_index, two_worst)

    draw(original_max_index, original_best, two_max_index, two_best, x, y, z1, z2)
    draw(original_min_index, original_worst, two_min_index, two_worst, x, y, z1, z2)
    draw_one(two_max_index, two_best, x, y, z2)
    draw_one(two_min_index, two_worst, x, y, z2)


