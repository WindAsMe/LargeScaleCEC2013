from __future__ import print_function
import numpy as np
import geatpy as ea
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from TestBenchmark.SparseForBenchmark import create_data
from TestBenchmark.SparseForBenchmark import create_result
import TestBenchmark.benchmark
from sklearn.metrics import r2_score


def min_max_normalization(array):
    distance = []
    result = []
    minValue = min(array)
    maxValue = max(array)
    for a in array:
        a = (a - minValue) / (maxValue - minValue)
        result.append(a)

    for i in range(1, len(array), 2):
        distance.append(result[i] - result[i - 1])
    print('distance: ', distance)
    return result


def RegressionFunction(degree, f):
    train_data = create_data(1000, 5)
    test_data = create_data(100, 5)

    train_result = create_result(train_data, f)
    test_result = create_result(test_data, f)

    poly_reg = PolynomialFeatures(degree=degree)
    train_data_poly = poly_reg.fit_transform(train_data)
    test_data_poly = poly_reg.fit_transform(test_data)
    reg = linear_model.Lasso()
    reg.fit(train_data_poly, train_result)

    trainR2 = r2_score(train_result, reg.predict(train_data_poly))
    testR2 = r2_score(test_result, reg.predict(test_data_poly))
    print('trainR2: ', trainR2, ' testR2: ', testR2)
    return reg


def aim_Griewank_original(Phen):
    x0 = Phen[:, [0]]
    x1 = Phen[:, [1]]
    x2 = Phen[:, [2]]
    x3 = Phen[:, [3]]
    x4 = Phen[:, [4]]
    part1 = x0 ** 2 / 4000 + x1 ** 2 / 4000 + x2 ** 2 / 4000 + x3 ** 2 / 4000 + x4 ** 2 / 4000
    part2 = np.cos(x0 / np.sqrt(1)) * np.cos(x1 / np.sqrt(2)) * np.cos(x2 / np.sqrt(3)) * np.cos(x3 / np.sqrt(4)) * np.cos(x4 / np.sqrt(5))
    return part1 - part2 + 1


def aim_Griewank_two(Phen, reg):
    poly_reg = PolynomialFeatures(degree=2)
    Phen_poly = poly_reg.fit_transform(Phen)
    array = np.array([reg.predict(Phen_poly)]).T
    return array


def aim_Griewank_three(Phen, reg):
    poly_reg = PolynomialFeatures(degree=3)
    Phen_poly = poly_reg.fit_transform(Phen)
    array = np.array([reg.predict(Phen_poly)]).T
    return array


def aim_Schwefel_original(Phen):
    x0 = Phen[:, [0]]
    x1 = Phen[:, [1]]
    x2 = Phen[:, [2]]
    x3 = Phen[:, [3]]
    x4 = Phen[:, [4]]
    return 418.9829 * 5 - (x0 * np.sin(np.sqrt(abs(x0))) + x1 * np.sin(np.sqrt(abs(x1))) +
                           x2 * np.sin(np.sqrt(abs(x2))) + x3 * np.sin(np.sqrt(abs(x3))) +
                           x4 * np.sin(np.sqrt(abs(x4))))


def aim_Schwefel_two(Phen, reg):
    poly_reg = PolynomialFeatures(degree=2)
    Phen_poly = poly_reg.fit_transform(Phen)
    array = np.array([reg.predict(Phen_poly)]).T
    return array


def aim_Schwefel_three(Phen, reg):
    poly_reg = PolynomialFeatures(degree=3)
    Phen_poly = poly_reg.fit_transform(Phen)
    array = np.array([reg.predict(Phen_poly)]).T
    return array


def aim_Rosenbrock_original(Phen):
    x0 = Phen[:, [0]]
    x1 = Phen[:, [1]]
    x2 = Phen[:, [2]]
    x3 = Phen[:, [3]]
    x4 = Phen[:, [4]]
    return (100 * (x0 ** 2 - x1) ** 2 + (x0 - 1) ** 2) + (100 * (x1 ** 2 - x2) ** 2 + (x1 - 1) ** 2) + \
           (100 * (x2 ** 2 - x3) ** 2 + (x2 - 1) ** 2) + (100 * (x3 ** 2 - x4) ** 2 + (x3 - 1) ** 2)


def aim_Rosenbrock_two(Phen, reg):
    poly_reg = PolynomialFeatures(degree=2)
    Phen_poly = poly_reg.fit_transform(Phen)
    array = np.array([reg.predict(Phen_poly)]).T
    return array


def aim_Rosenbrock_three(Phen, reg):
    poly_reg = PolynomialFeatures(degree=3)
    Phen_poly = poly_reg.fit_transform(Phen)
    array = np.array([reg.predict(Phen_poly)]).T
    return array


def aim_Rastrigin_original(Phen):
    x0 = Phen[:, [0]]
    x1 = Phen[:, [1]]
    x2 = Phen[:, [2]]
    x3 = Phen[:, [3]]
    x4 = Phen[:, [4]]
    return (x0 ** 2 - 10 * np.cos(2 * np.pi * x0) + 10) + (x1 ** 2 - 10 * np.cos(2 * np.pi * x1) + 10) + \
           (x2 ** 2 - 10 * np.cos(2 * np.pi * x2) + 10) + (x3 ** 2 - 10 * np.cos(2 * np.pi * x3) + 10) + \
           (x4 ** 2 - 10 * np.cos(2 * np.pi * x4) + 10)


def aim_Rastrigin_two(Phen, reg):
    poly_reg = PolynomialFeatures(degree=2)
    Phen_poly = poly_reg.fit_transform(Phen)
    array = np.array([reg.predict(Phen_poly)]).T
    return array


def aim_Rastrigin_three(Phen, reg):
    poly_reg = PolynomialFeatures(degree=3)
    Phen_poly = poly_reg.fit_transform(Phen)
    array = np.array([reg.predict(Phen_poly)]).T
    return array


def aim_Ackley_original(Phen):
    x0 = Phen[:, [0]]
    x1 = Phen[:, [1]]
    x2 = Phen[:, [2]]
    x3 = Phen[:, [3]]
    x4 = Phen[:, [4]]
    part1 = -0.2 * np.sqrt(1 / 5 * (x0 ** 2 + x1 ** 2 + x2 ** 2 + x3 ** 2 + x4 ** 2))
    part2 = 1 / 5 * (np.cos(2 * np.pi * x0) + np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2) + np.cos(2 * np.pi * x3) + np.cos(2 * np.pi * x4))
    return -20 * np.exp(part1) - np.exp(part2) + 20 + np.e


def aim_Ackley_two(Phen, reg):
    poly_reg = PolynomialFeatures(degree=2)
    Phen_poly = poly_reg.fit_transform(Phen)
    array = np.array([reg.predict(Phen_poly)]).T
    return array


def aim_Ackley_three(Phen, reg):
    poly_reg = PolynomialFeatures(degree=3)
    Phen_poly = poly_reg.fit_transform(Phen)
    array = np.array([reg.predict(Phen_poly)]).T
    return array


def OptimizationForOriginal(aim, name, maxormin):

    string = ""
    if maxormin == 1:
        string = "find minimum"
    if maxormin == -1:
        string = "find maximum"
    print(name)
    print(string)
    x1 = [-10, 10]
    x2 = [-10, 10]
    x3 = [-10, 10]
    x4 = [-10, 10]
    x5 = [-10, 10]

    b1 = [1, 1]
    b2 = [1, 1]
    b3 = [1, 1]
    b4 = [1, 1]
    b5 = [1, 1]
    ranges = np.vstack([x1, x2, x3, x4, x5]).T
    borders = np.vstack([b1, b2, b3, b4, b5]).T
    varTypes = np.array([0, 0, 0, 0, 0])
    Encoding = 'BG'
    codes = [1, 1, 1, 1, 1]
    precisions = [6, 6, 6, 6, 6]
    scales = [0, 0, 0, 0, 0]
    FieldD = ea.crtfld(Encoding, varTypes, ranges, borders, precisions, codes, scales)

    NIND = 20
    MAXGEN = 1000
    maxormins = [maxormin]
    selectStyle = 'sus'
    recStyle = 'xovdp'
    mutStyle = 'mutbin'
    Lind = int(np.sum(FieldD[0, :]))
    pc = 0.9
    pm = 1 / Lind
    obj_trace = np.zeros((MAXGEN, 2))
    var_trace = np.zeros((MAXGEN, Lind))

    # start_time = time.time()
    Chrom = ea.crtpc(Encoding, NIND, FieldD)
    variable = ea.bs2real(Chrom, FieldD)
    ObjV = aim(variable)
    best_ind = np.argmin(ObjV)
    for gen in range(MAXGEN):
        FitnV = ea.ranking(maxormins * ObjV)
        SelCh = Chrom[ea.selecting(selectStyle, FitnV, NIND - 1), :]
        SelCh = ea.recombin(recStyle, SelCh, pc)
        SelCh = ea.mutate(mutStyle, Encoding, SelCh, pm)
        Chrom = np.vstack([Chrom[best_ind, :], SelCh])
        Phen = ea.bs2real(Chrom, FieldD)
        ObjV = aim(Phen)
        best_ind = np.argmin(ObjV)
        obj_trace[gen, 0] = np.sum(ObjV) / ObjV.shape[0]
        obj_trace[gen, 1] = ObjV[best_ind]
        var_trace[gen, :] = Chrom[best_ind, :]
    # end_time = time.time()
    ea.trcplot(obj_trace, [['种群个体平均目标函数值', '种群最优个体目标函数值']])

    best_gen = np.argmin(obj_trace[:, [1]])
    print("The best Individual: ", obj_trace[best_gen, 1])
    print('--------------------------------------')
    # variable = ea.bs2real(var_trace[[best_gen], :], FieldD)
    # print('最优解的决策变量值为：')
    # for i in range(variable.shape[1]):
    #     print('x' + str(i) + '=', variable[0, i])
    # print('用时：', end_time - start_time)
    return obj_trace[best_gen, 1]


def OptimizationForRegression(aim, name, reg, maxormin):

    string = ""
    if maxormin == 1:
        string = "find minimum"
    if maxormin == -1:
        string = "find maximum"
    print(name)
    print(string)
    x1 = [-10, 10]
    x2 = [-10, 10]
    x3 = [-10, 10]
    x4 = [-10, 10]
    x5 = [-10, 10]

    b1 = [1, 1]
    b2 = [1, 1]
    b3 = [1, 1]
    b4 = [1, 1]
    b5 = [1, 1]
    ranges = np.vstack([x1, x2, x3, x4, x5]).T
    borders = np.vstack([b1, b2, b3, b4, b5]).T
    varTypes = np.array([0, 0, 0, 0, 0])
    Encoding = 'BG'
    codes = [1, 1, 1, 1, 1]
    precisions = [6, 6, 6, 6, 6]
    scales = [0, 0, 0, 0, 0]
    FieldD = ea.crtfld(Encoding, varTypes, ranges, borders, precisions, codes, scales)

    NIND = 20
    MAXGEN = 1000
    maxormins = [maxormin]
    selectStyle = 'sus'
    recStyle = 'xovdp'
    mutStyle = 'mutbin'
    Lind = int(np.sum(FieldD[0, :]))
    pc = 0.9
    pm = 1 / Lind
    obj_trace = np.zeros((MAXGEN, 2))
    var_trace = np.zeros((MAXGEN, Lind))

    # start_time = time.time()
    Chrom = ea.crtpc(Encoding, NIND, FieldD)
    variable = ea.bs2real(Chrom, FieldD)
    ObjV = aim(variable, reg)
    best_ind = np.argmin(ObjV)
    for gen in range(MAXGEN):
        FitnV = ea.ranking(maxormins * ObjV)
        SelCh = Chrom[ea.selecting(selectStyle, FitnV, NIND-1), :]
        SelCh = ea.recombin(recStyle, SelCh, pc)
        SelCh = ea.mutate(mutStyle, Encoding, SelCh, pm)
        Chrom = np.vstack([Chrom[best_ind, :], SelCh])
        Phen = ea.bs2real(Chrom, FieldD)
        ObjV = aim(Phen, reg)
        best_ind = np.argmin(ObjV)
        obj_trace[gen, 0] = np.sum(ObjV)/ObjV.shape[0]
        obj_trace[gen, 1] = ObjV[best_ind]
        var_trace[gen, :] = Chrom[best_ind, :]
    # end_time = time.time()
    ea.trcplot(obj_trace, [['种群个体平均目标函数值', '种群最优个体目标函数值']])

    best_gen = np.argmin(obj_trace[:, [1]])
    print('The best Individual：', obj_trace[best_gen, 1])
    print('--------------------------------------')
    # variable = ea.bs2real(var_trace[[best_gen], :], FieldD)
    # print('最优解的决策变量值为：')
    # for i in range(variable.shape[1]):
    #     print('x' + str(i) + '=', variable[0, i])
    # print('用时：', end_time - start_time)
    return obj_trace[best_gen, 1]


if __name__ == '__main__':

    GriewankOriginal = [OptimizationForOriginal(aim_Griewank_original, 'Griewank function originally', 1),
                        OptimizationForOriginal(aim_Griewank_original, 'Griewank function originally', -1)]

    regGriewankTwo = RegressionFunction(2, TestBenchmark.benchmark.Griewank)
    regGriewankThree = RegressionFunction(3, TestBenchmark.benchmark.Griewank)
    GriewankTwo = [OptimizationForRegression(aim_Griewank_two, 'Griewank function degree = 2', regGriewankTwo, 1),
                   OptimizationForRegression(aim_Griewank_two, 'Griewank function degree = 2', regGriewankTwo, -1)]
    GriewankThree = [OptimizationForRegression(aim_Griewank_three, 'Griewank function degree = 3', regGriewankThree, 1),
                     OptimizationForRegression(aim_Griewank_three, 'Griewank function degree = 3', regGriewankThree, -1)]

    print()
    regSchwefelTwo = RegressionFunction(2, TestBenchmark.benchmark.Schwefel)
    regSchwefelThree = RegressionFunction(3, TestBenchmark.benchmark.Schwefel)
    SchwefelOriginal = [OptimizationForOriginal(aim_Schwefel_original, 'Schwefel function originally', 1),
                        OptimizationForOriginal(aim_Schwefel_original, 'Schwefel function originally', -1)]
    SchwefelTwo = [OptimizationForRegression(aim_Schwefel_two, 'Schwefel function degree = 2', regSchwefelTwo, 1),
                    OptimizationForRegression(aim_Schwefel_two, 'Schwefel function degree = 2', regSchwefelTwo, -1)]
    SchwefelThree = [OptimizationForRegression(aim_Schwefel_three, 'Schwefel function degree = 3', regSchwefelThree, 1),
                        OptimizationForRegression(aim_Schwefel_three, 'Schwefel function degree = 3', regSchwefelThree, -1)]

    print()
    regRosenbrockTwo = RegressionFunction(2, TestBenchmark.benchmark.Rosenbrock)
    regRosenbrockThree = RegressionFunction(3, TestBenchmark.benchmark.Rosenbrock)
    RosenbrockOriginal = [OptimizationForOriginal(aim_Rosenbrock_original, 'Rosenbrock function originally', 1),
                            OptimizationForOriginal(aim_Rosenbrock_original, 'Rosenbrock function originally', -1)]
    RosenbrockTwo = [OptimizationForRegression(aim_Rosenbrock_two, 'Rosenbrock function degree = 2', regRosenbrockTwo, 1),
                        OptimizationForRegression(aim_Rosenbrock_two, 'Rosenbrock function degree = 2', regRosenbrockTwo, -1)]
    RosenbrockThree = [OptimizationForRegression(aim_Rosenbrock_three, 'Rosenbrock function degree = 3', regRosenbrockThree, 1),
                        OptimizationForRegression(aim_Rosenbrock_three, 'Rosenbrock function degree = 3', regRosenbrockThree, -1)]

    print()
    regRastriginTwo = RegressionFunction(2, TestBenchmark.benchmark.Rastrigin)
    regRastriginThree = RegressionFunction(3, TestBenchmark.benchmark.Rastrigin)
    RastriginOriginal = [OptimizationForOriginal(aim_Rastrigin_original, 'Rastrigin function originally', 1),
                            OptimizationForOriginal(aim_Rastrigin_original, 'Rastrigin function originally', -1)]
    RastriginTwo = [OptimizationForRegression(aim_Rastrigin_two, 'Rastrigin function degree = 2', regRastriginTwo, 1),
                    OptimizationForRegression(aim_Rastrigin_two, 'Rastrigin function degree = 2', regRastriginTwo, -1)]
    RastriginThree = [OptimizationForRegression(aim_Rastrigin_three, 'Rastrigin function degree = 3', regRastriginThree, 1),
                        OptimizationForRegression(aim_Rastrigin_three, 'Rastrigin function degree = 3', regRastriginThree, -1)]

    print()
    regAckleyTwo = RegressionFunction(2, TestBenchmark.benchmark.Ackley)
    regAckleyThree = RegressionFunction(3, TestBenchmark.benchmark.Ackley)
    AckleyOriginal = [OptimizationForOriginal(aim_Ackley_original, 'Ackley function originally', 1),
                        OptimizationForOriginal(aim_Ackley_original, 'Ackley function originally', -1)]
    AckleyTwo = [OptimizationForRegression(aim_Ackley_two, 'Ackley function degree = 2', regAckleyTwo, 1),
                    OptimizationForRegression(aim_Ackley_two, 'Ackley function degree = 2', regAckleyTwo, -1)]
    AckleyThree = [OptimizationForRegression(aim_Ackley_three, 'Ackley function degree = 3', regAckleyThree, 1),
                    OptimizationForRegression(aim_Ackley_three, 'Ackley function degree = 3', regAckleyThree, -1)]

    print('--------------------------------------------------')
    print(GriewankOriginal, GriewankTwo, GriewankThree)
    print(SchwefelOriginal, SchwefelTwo, SchwefelThree)
    print(RosenbrockOriginal, RosenbrockTwo, RosenbrockThree)
    print(RastriginOriginal, RastriginTwo, RastriginThree)
    print(AckleyOriginal, AckleyTwo, AckleyThree)

    print()

    print('Normalization:')
    print('Griewank')
    GriewankArray = np.array(GriewankOriginal + GriewankTwo + GriewankThree)

    Griewank_min_max = min_max_normalization(GriewankArray)
    print('max_min:', Griewank_min_max)

    print('Schwefel')
    SchwefelArray = np.array(SchwefelOriginal + SchwefelTwo + SchwefelThree)
    Schwefel_min_max = min_max_normalization(SchwefelArray)
    print('max_min:', Schwefel_min_max)

    print('Rosenbrock')
    RosenbrockArray = np.array(RosenbrockOriginal + RosenbrockTwo + RosenbrockThree)
    Rosenbrock_min_max = min_max_normalization(RosenbrockArray)
    print('max_min:', Rosenbrock_min_max)

    print('Rastrigin')
    RastriginArray = np.array(RastriginOriginal + RastriginTwo + RastriginThree)
    Rastrigin_min_max = min_max_normalization(RastriginArray)
    print('max_min:', Rastrigin_min_max)

    print('Ackley')
    AckleyArray = np.array(AckleyOriginal + AckleyTwo + AckleyThree)
    Ackley_min_max = min_max_normalization(AckleyArray)
    print('max_min:', Ackley_min_max)
