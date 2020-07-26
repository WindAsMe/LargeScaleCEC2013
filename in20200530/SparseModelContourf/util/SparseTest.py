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


def Regression(degree, train_data, train_result):

    poly_reg = PolynomialFeatures(degree=degree)
    train_data_poly = poly_reg.fit_transform(train_data)

    reg = linear_model.Lasso()
    reg.fit(train_data_poly, train_result)

    return reg


def SparseModeling(feature_size, function, degree):
    train_data = create_data(500, feature_size)
    train_label = create_result(train_data, function)
    reg = Regression(degree, train_data, train_label)
    return reg
