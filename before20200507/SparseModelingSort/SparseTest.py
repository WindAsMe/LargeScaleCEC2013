from sklearn import linear_model
import random
from sklearn.preprocessing import PolynomialFeatures


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


def get_feature_name(number):
    result = []
    for i in range(number):
        result.append('x' + str(i))
    return result


def find_valid_coefficients(feature_name, coef):
    max_value = max(abs(coef))
    # print("max value: ", max_value)
    valid_feature_name = []
    valid_coef = []
    valid_index = []
    for i in range(len(coef)):
        if abs(coef[i]) > 0.001 and abs(coef[i]) > max_value * 0.05:
            valid_feature_name.append(feature_name[i])
            valid_coef.append(coef[i])
            valid_index.append(i)
    print("All valid coefficients len: ", len(valid_coef))
    print("Valid coefficients len: ", len(valid_feature_name))

    separable = 0
    separable_name = []
    for c in valid_feature_name:
        if isSeparable(c):
            separable_name.append(c)
            separable += 1
    print("DimensionReductionForSparse number: ", separable)
    return valid_index


def isSeparable(str):
    str_c = str.replace(' ', '')
    return len(str_c) == len(str)


# def adaptive_lasso_coef(coef, train_data_poly):
#     max_value = max(abs(coef))
#     for i in range(len(coef)):
#         if abs(coef[i]) > 0.001 and abs(coef[i]) > max_value * 0.05:
#             train_data_poly[:, i] = 1 / coef[i] * train_data_poly[:, i]
#         else:
#             train_data_poly[:, i] = 10000 * train_data_poly[:, i]
#     return train_data_poly


def Regression(degree, train_data, train_result, feature_size):

    poly_reg = PolynomialFeatures(degree=degree)
    train_data_poly = poly_reg.fit_transform(train_data)

    # Tag the vars name with combination
    # feature_name = poly_reg.get_feature_names(input_features=get_feature_name(feature_size))

    reg = linear_model.Lasso()
    reg.fit(train_data_poly, train_result)
    # valid_index = find_valid_coefficients(feature_name, reg.coef_)
    # train_data_poly_adaptive = adaptive_lasso_coef(reg.coef_, train_data_poly)

    # reg.fit(train_data_poly_adaptive, train_result)

    # print(reg.coef_)

    # for i in range(len(reg.coef_)):
    #     if i not in valid_index:
    #         reg.coef_[i] = 0

    return reg


def SparseModeling(feature_size, function, degree):
    train_data = create_data(50000, feature_size)
    train_label = create_result(train_data, function)
    reg = Regression(degree, train_data, train_label, feature_size)
    return reg
