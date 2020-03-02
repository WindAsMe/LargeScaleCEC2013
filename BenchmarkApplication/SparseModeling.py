from TestBenchmark import benchmark
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import random
import numpy as np
import geatpy as ea
import heapq
from BenchmarkApplication import aims
from sklearn.metrics import r2_score


def find_nlargest_index(coef, feature, topk=3):
    coef_abs = abs(coef)
    nlargest = heapq.nlargest(topk, coef_abs)
    result = []
    for largest in nlargest:
        for i in range(0, len(coef)):
            if abs(coef[i]) == largest:
                result.append(i)
    print(nlargest)
    print(result)
    pairs = []
    for i in result:
        pairs.append([coef[i], feature[i]])
    return pairs


def RegressionFunction(degree, f, dimension, features):
    train_data = create_data(10000, dimension)

    # To prevent overfitting
    test_data = create_data(100, dimension)
    train_result = create_result(train_data, f)
    test_result = create_result(test_data, f)

    poly_reg = PolynomialFeatures(degree=degree)
    train_data_poly = poly_reg.fit_transform(train_data)
    test_data_poly = poly_reg.fit_transform(test_data)

    reg = linear_model.Lasso()
    reg.fit(train_data_poly, train_result)
    feature_name = poly_reg.get_feature_names(input_features=features)
    print('test R2: ', r2_score(test_result, reg.predict(test_data_poly)))
    print('When degree = ', degree, ', the coefficients of factors')

    # save the factors like x0^2, x1^2
    found_feature = [1]
    temp_degree = degree
    while temp_degree > 1:
        for feature in features:
            found_feature.append(feature + '^' + str(temp_degree))
        temp_degree -= 1
    for feature in features:
        found_feature.append(feature)

    # save the corresponding index
    separable_index = [1]
    unseparable_index = []
    for i in range(len(feature_name)):
        flag = 0
        for element in found_feature:
            if feature_name[i] == element:
                separable_index.append(i)
                flag = 1
                break
        if flag == 0:
            unseparable_index.append(i)

    separable_element = []
    unseparable_element = []
    for index in separable_index:
        separable_element.append(reg.coef_[index])
    for index in unseparable_index:
        unseparable_element.append(reg.coef_[index])

    separable_index_not_zero = []
    unseparable_index_not_zero = []
    separable_element_not_zero = []
    unseparable_element_not_zero = []

    separable_element_unignored = []
    unseparable_element_unignored = []

    for index in separable_index:
        if reg.coef_[index] != 0:
            separable_index_not_zero.append(index)
            separable_element_not_zero.append(reg.coef_[index])
    for index in unseparable_index:
        if reg.coef_[index] != 0:
            unseparable_index_not_zero.append(index)
            unseparable_element_not_zero.append(reg.coef_[index])

    separable_max = 0
    unseparable_max = 0
    temp_separable_index_not_zero = []
    temp_unseparable_index_not_zero = []
    for element in separable_index_not_zero:
        temp_separable_index_not_zero.append(abs(element))
    for element in unseparable_element_not_zero:
        temp_unseparable_index_not_zero.append(abs(element))

    if len(separable_index_not_zero) > 0:
        separable_max = max(separable_index_not_zero) * 0.05
    if len(unseparable_element_not_zero) > 0:
        unseparable_max = max(unseparable_index_not_zero) * 0.05

    for index in range(len(separable_index_not_zero)):
        if temp_separable_index_not_zero[index] > 0.001 and temp_separable_index_not_zero[index] > separable_max:
            separable_element_unignored.append(separable_index_not_zero[index])
    for index in range(len(unseparable_index_not_zero)):
        if temp_unseparable_index_not_zero[index] > 0.001 and temp_unseparable_index_not_zero[index] > unseparable_max:
            unseparable_element_unignored.append(unseparable_index_not_zero[index])

    print('total feature: ', len(feature_name))
    print('the separable features which are not 0: ', len(separable_index_not_zero))
    print('the unseparable features which are not 0: ', len(unseparable_index_not_zero))
    print('the separable features which can\'t be ignored: ', len(separable_element_unignored))
    print('the unseparable features which can\'t be ignored:', len(unseparable_element_unignored))

    # print(find_nlargest_index(reg.coef_, feature_name, 50))
    print('---------------------------------')
    return reg


def OptimizationForOriginal(aim, name, maxormin, dimension):

    string = ""
    if maxormin == 1:
        string = "find minimum"
    if maxormin == -1:
        string = "find maximum"
    print(name)
    print(string)

    range_item = []
    border_item = []
    varType_item = []
    code_item = []
    precision_item = []
    scale_item = []
    for i in range(0, dimension):
        range_item.append([-10, 10])
        border_item.append([1, 1])
        varType_item.append(0)
        code_item.append(1)
        precision_item.append(6)
        scale_item.append(0)
    ranges = np.vstack(range_item).T
    borders = np.vstack(border_item).T
    varTypes = np.array(varType_item)
    Encoding = 'BG'
    codes = code_item
    precisions = precision_item
    scales = scale_item
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
    ObjV = aim(variable, dimension)
    best_ind = np.argmin(ObjV)
    for gen in range(MAXGEN):
        FitnV = ea.ranking(maxormins * ObjV)
        SelCh = Chrom[ea.selecting(selectStyle, FitnV, NIND - 1), :]
        SelCh = ea.recombin(recStyle, SelCh, pc)
        SelCh = ea.mutate(mutStyle, Encoding, SelCh, pm)
        Chrom = np.vstack([Chrom[best_ind, :], SelCh])
        Phen = ea.bs2real(Chrom, FieldD)
        ObjV = aim(Phen, dimension)
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
    # print('For the optimization factors:')
    index = []
    for i in range(variable.shape[1]):
        # print('x' + str(i) + '=', variable[0, i])
        index.append(variable[0, i])
    # print('用时：', end_time - start_time)
    return obj_trace[best_gen, 1], index


def OptimizationForRegression(aim, name, reg, maxormin, dimension):

    string = ""
    if maxormin == 1:
        string = "find minimum"
    if maxormin == -1:
        string = "find maximum"
    print(name)
    print(string)

    range_item = []
    border_item = []
    varType_item = []
    code_item = []
    precision_item = []
    scale_item = []
    for i in range(0, dimension):
        range_item.append([-10, 10])
        border_item.append([1, 1])
        varType_item.append(0)
        code_item.append(1)
        precision_item.append(6)
        scale_item.append(0)
    ranges = np.vstack(range_item).T
    borders = np.vstack(border_item).T
    varTypes = np.array(varType_item)
    Encoding = 'BG'
    codes = code_item
    precisions = precision_item
    scales = scale_item
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
    # print('For the optimization factors:')
    index = []
    for i in range(variable.shape[1]):
        # print('x' + str(i) + '=', variable[0, i])
        index.append(variable[0, i])
    # print('用时：', end_time - start_time)
    return obj_trace[best_gen, 1], index


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


def find_same_factors_index(sub_list, original_list):
    # The length of result is equal with sub_list
    result = []
    for element in sub_list:
        for index, factor in enumerate(original_list):
            if element == factor:
                result.append(index)
                continue
    return result


def create_feature(feature_num):
    features = []
    for i in range(feature_num):
        features.append('x' + str(i))
    return features


if __name__ == '__main__':
    dimension = 100
    feature_num = 100
    features = create_feature(feature_num)

    name = 'Griewank'
    function_original = aims.aim_Griewank_original
    function_two = aims.aim_Griewank_two
    function_three = aims.aim_Griewank_three
    benchmark_function = benchmark.Griewank

    # min_individual_original, min_index_original = OptimizationForOriginal(function_original, name +
    #                                                                                         ' function originally',
    #                                                                                         1, dimension)
    # max_individual_original, max_index_original = OptimizationForOriginal(function_original, name +
    #                                                                                         ' function originally',
    #                                                                                         -1, dimension)

    regTwo = RegressionFunction(2, benchmark_function, dimension, features)
    regThree = RegressionFunction(3, benchmark_function, dimension, features)
    # min_individual_Griewank_two, min_index_two = OptimizationForRegression(function_two, name +
    #                                                                                     ' function degree = 2',
    #                                                                                     regTwo, 1, dimension)
    # max_individual_two, max_index_two = OptimizationForRegression(function_two, name +
    #                                                                                     ' function degree = 2',
    #                                                                                     regTwo, -1, dimension)
    # min_individual_three, min_index_three = OptimizationForRegression(function_three, name +
    #                                                                                     ' function degree = 3',
    #                                                                                     regThree, 1, dimension)
    # max_individual_three, max_index_three = OptimizationForRegression(function_three, name +
    #                                                                                     ' function degree = 3',
    #                                                                                     regThree, -1, dimension)
    #
    # print(stats.wilcoxon(max_index_original, max_index_two, correction=True, alternative='greater'))
    # print(stats.wilcoxon(min_index_original, min_index_two, correction=True, alternative='greater'))
    # print(stats.wilcoxon(max_index_original, max_index_three, correction=True, alternative='greater'))
    # print(stats.wilcoxon(min_index_original, min_index_three, correction=True, alternative='greater'))

