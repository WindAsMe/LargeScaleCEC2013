from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from in20201030.DimensionReductionForSparse.util import help
import numpy as np


def get_feature_name(number):
    result = []
    for i in range(number):
        result.append(str(i))
    return result


def isSeparable(str):
    str_c = str.replace(' ', '')
    return len(str_c) == len(str)


def is_zero(coef):
    num = 0
    for i in coef:
        if i == 0:
            num += 1
    return num


def Regression(train_size, part, scale_range, benchmark):

    poly_reg = PolynomialFeatures(degree=2)
    train_data = help.create_model_data(train_size, part, scale_range)
    train_label = help.create_result(train_data, benchmark)
    reg_Lasso = linear_model.Lasso()
    true_train_data = train_data[:, (part-1)*100:part*100]
    true_train_data_poly = poly_reg.fit_transform(true_train_data)
    reg_Lasso.fit(true_train_data_poly, train_label)
    feature_names = poly_reg.get_feature_names(input_features=get_feature_name(100))
    flag = max(abs(reg_Lasso.coef_))
    print(reg_Lasso.coef_)
    valid_feature = []
    valid_coef = []
    for i in range(len(reg_Lasso.coef_)):
        if abs(reg_Lasso.coef_[i]) > 0.01 and abs(reg_Lasso.coef_[i]) > flag * 0.01:
            valid_feature.append(feature_names[i])
            valid_coef.append(reg_Lasso.coef_[i])
            continue
        else:
            reg_Lasso.coef_[i] = 0
    print(valid_feature)
    print(valid_coef)
    print('Sparse model valid coef: ', len(reg_Lasso.coef_) - is_zero(reg_Lasso.coef_), 'intercept: ', reg_Lasso.intercept_)
    return reg_Lasso, feature_names
