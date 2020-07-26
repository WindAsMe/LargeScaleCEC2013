import random
from before20200507.preTest import formula
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score


# train data: scale=100
# test data: scale=50
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


def simulate_where(original_list, element=''):
    for i in range(0, len(original_list)):
        if element == original_list[i]:
            return i


def Regression(degree, train_data, train_result, test_data, test_result, flag):

    poly_reg = PolynomialFeatures(degree=degree)
    train_data_poly = poly_reg.fit_transform(train_data)
    test_data_poly = poly_reg.fit_transform(test_data)

    # Tag the vars name with combination
    feature_name = poly_reg.get_feature_names(input_features=['x0', 'x1', 'x2', 'x3', 'x4'])

    reg = linear_model.Lasso()
    reg.fit(train_data_poly, train_result)
    train_predict = reg.predict(train_data_poly)
    test_predict = reg.predict(test_data_poly)

    # The following is the result
    # trainMSE = mean_squared_error(train_result, train_predict)
    # testMSE = mean_squared_error(test_result, test_predict)

    trainR2 = r2_score(train_result, train_predict)
    testR2 = r2_score(test_result, test_predict)
    print("The fit function's degree = ", degree)
    print("train data R2: ", trainR2, "    test data R2: ", testR2)
    if flag == 1:
        print("The coefficients of previous function: ", [1, 3, 2, a, b])
        if degree == 1:
            print("The coefficients of fit function: ",
                  [reg.coef_[simulate_where(feature_name, 'x0')], reg.coef_[simulate_where(feature_name, 'x1')],
                   reg.coef_[simulate_where(feature_name, 'x2')], reg.coef_[simulate_where(feature_name, 'x3')],
                   reg.coef_[simulate_where(feature_name, 'x4')]])
        else:
            print(len(reg.coef_))
            print("The coefficients of fit function: ",
                  [reg.coef_[simulate_where(feature_name, 'x0^2')], reg.coef_[simulate_where(feature_name, 'x1^2')],
                   reg.coef_[simulate_where(feature_name, 'x1 x2')], reg.coef_[simulate_where(feature_name, 'x3^2')],
                   reg.coef_[simulate_where(feature_name, 'x4')]])

    if flag == 2:
        print("The coefficients of previous function: ", [1, 3, 2, a, b])
        if degree == 1:
            print("The coefficients of fit function: ",
                  [reg.coef_[simulate_where(feature_name, 'x0')], reg.coef_[simulate_where(feature_name, 'x1')],
                   reg.coef_[simulate_where(feature_name, 'x2')], reg.coef_[simulate_where(feature_name, 'x3')],
                   reg.coef_[simulate_where(feature_name, 'x4')], reg.coef_[simulate_where(feature_name, '1')]])
        else:
            # print(reg.coef_)
            print("The coefficients of fit function: ",
                  [reg.coef_[simulate_where(feature_name, 'x0^2')], reg.coef_[simulate_where(feature_name, 'x1^2')],
                   reg.coef_[simulate_where(feature_name, 'x1 x2')], reg.coef_[simulate_where(feature_name, 'x3^2')],
                   reg.coef_[simulate_where(feature_name, 'x4^2')], reg.coef_[simulate_where(feature_name, '1')]])

    print("---------------------------------------------------------------------------------")


if __name__ == '__main__':

    # a = random.randint(-5, 5)
    # b = random.randint(-5, 5)
    a = 5
    b = 2
    print("f1: purely polynomial function\n")
    train_data = create_data(100, 5)
    test_data = create_data(50, 5)
    train_result = create_result(train_data, formula.f)
    test_result = create_result(test_data, formula.f)
    Regression(1, train_data, train_result, test_data, test_result, 1)
    Regression(2, train_data, train_result, test_data, test_result, 1)
    Regression(3, train_data, train_result, test_data, test_result, 1)

    print()
    print("f2: polynomial function with Gaussian function\n")
    train_data = create_data(100, 5)
    test_data = create_data(50, 5)
    train_result = create_result(train_data, formula.f)
    test_result = create_result(test_data, formula.f)

    Regression(1, train_data, train_result, test_data, test_result, 2)
    Regression(2, train_data, train_result, test_data, test_result, 2)
    Regression(3, train_data, train_result, test_data, test_result, 2)



