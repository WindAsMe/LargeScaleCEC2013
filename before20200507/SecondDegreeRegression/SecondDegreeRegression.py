from before20200507.TestBenchmark import benchmark
from sklearn import linear_model
import random
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score


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


def Regression(degree, train_data, train_result, test_data, test_result):

    poly_reg = PolynomialFeatures(degree=degree)
    train_data_poly = poly_reg.fit_transform(train_data)
    test_data_poly = poly_reg.fit_transform(test_data)

    # Tag the vars name with combination
    # feature_name = poly_reg.get_feature_names(input_features=['x0', 'x1', 'x2', 'x3', 'x4'])

    reg = linear_model.Lasso()
    reg.fit(train_data_poly, train_result)
    train_predict = reg.predict(train_data_poly)
    test_predict = reg.predict(test_data_poly)

    trainR2 = r2_score(train_result, train_predict)
    testR2 = r2_score(test_result, test_predict)
    print("The fit function's degree = ", degree)
    print("train data R2: ", trainR2, "    test data R2: ", testR2)

    print("The number of coef: ", len(reg.coef_))


if __name__ == '__main__':
    train_data = create_data(50000, 100)
    train_label = create_result(train_data, benchmark.Griewank)
    test_data = create_data(500, 100)
    test_label = create_result(test_data, benchmark.Griewank)
    Regression(2, train_data, train_label, test_data, test_label)
