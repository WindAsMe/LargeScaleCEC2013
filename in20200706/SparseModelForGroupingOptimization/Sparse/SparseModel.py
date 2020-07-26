from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model


def get_feature_name(number):
    result = []
    for i in range(number):
        result.append('x' + str(i))
    return result


def find_valid_coefficients(feature_name, coef):
    max_value = max(abs(coef))

    valid_feature_name = []
    valid_coef = []
    valid_index = []
    for i in range(len(coef)):
        if abs(coef[i]) > 0.01 and abs(coef[i]) > max_value * 0.01:
            valid_feature_name.append(feature_name[i])
            valid_coef.append(coef[i])
            valid_index.append(i)
    # print("Valid coefficients len: ", len(valid_feature_name))

    return valid_index


def isSeparable(str):
    str_c = str.replace(' ', '')
    return len(str_c) == len(str)


def is_zero(coef):
    num = 0
    for i in coef:
        if i == 0:
            num += 1
    return num


def not_zero_feature(coef, feature_names):
    not_zero_coef = []
    not_zero_feature = []
    for index in range(len(coef)):
        if coef[index] != 0:
            not_zero_coef.append(coef[index])
            not_zero_feature.append(feature_names[index])
    print(not_zero_coef)
    print(not_zero_feature)


def Regression(degree, train_data, train_result, feature_size):

    poly_reg = PolynomialFeatures(degree=degree)
    train_data_poly = poly_reg.fit_transform(train_data)
    print(train_data_poly.shape)
    # Tag the vars name with combination
    feature_names = poly_reg.get_feature_names(input_features=get_feature_name(feature_size))
    print('feature names: ', len(feature_names))
    reg_Lasso = linear_model.Lasso()
    reg_Lasso.fit(train_data_poly, train_result)

    not_zero_feature(reg_Lasso.coef_, feature_names)

    valid_index = find_valid_coefficients(feature_names, reg_Lasso.coef_)
    for i in range(len(reg_Lasso.coef_)):
        if i not in valid_index:
            reg_Lasso.coef_[i] = 0

    print('Sparse model valid coef: ', len(reg_Lasso.coef_) - is_zero(reg_Lasso.coef_), 'intercept: ', reg_Lasso.intercept_)
    return reg_Lasso, feature_names

