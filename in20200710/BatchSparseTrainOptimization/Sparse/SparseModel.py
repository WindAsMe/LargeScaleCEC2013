from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import numpy as np
from in20200710.BatchSparseTrainOptimization.util import help, benchmark
from sklearn.preprocessing import StandardScaler


def get_feature_name(number):
    result = []
    for i in range(number):
        result.append(str(i))
    return result


def is_separable(str):
    str_c = str.replace(' ', '')
    return len(str_c) == len(str)


def is_zero(coef):
    num = 0
    for i in coef:
        if i == 0:
            num += 1
    return num


def Regression(degree, train_size, Func_Dim, mini_batch_size, scale_range, Func_Num):

    poly_reg = PolynomialFeatures(degree=degree)
    reg_Lasso = linear_model.SGDRegressor(learning_rate='optimal', penalty='l1', l1_ratio=1, loss='huber', tol=1e-2, average=True)
    train_data = help.create_data(train_size, Func_Dim, scale_range)

    times = int(train_size / mini_batch_size)

    print('Total times: ', times)
    for i in range(times):
        print('Sparse model build round ', i)
        partial_train_data = train_data[i*mini_batch_size:(i+1)*mini_batch_size, :]
        partial_train_label = help.create_result(partial_train_data, benchmark.f_evaluation, Func_Num)
        partial_train_data_poly = np.array(poly_reg.fit_transform(partial_train_data), dtype='float16')
        reg_Lasso.partial_fit(partial_train_data_poly, partial_train_label)
    feature_names = poly_reg.get_feature_names(input_features=get_feature_name(Func_Dim))

    flag = max(abs(reg_Lasso.coef_))
    print('max coef: ', flag)
    valid_feature = []
    valid_coef = []
    for i in range(len(reg_Lasso.coef_)):
        if abs(reg_Lasso.coef_[i]) > 0.1 and abs(reg_Lasso.coef_[i]) > flag * 0.2:
            valid_feature.append(feature_names[i])
            valid_coef.append(reg_Lasso.coef_[i])
            continue
        else:
            reg_Lasso.coef_[i] = 0
    print('Sparse model valid coef: ', len(reg_Lasso.coef_) - is_zero(reg_Lasso.coef_), 'intercept: ', reg_Lasso.intercept_)
    return reg_Lasso, feature_names
