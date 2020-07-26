from before20200507.TestBenchmark import benchmark
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import random
import heapq


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


def Regression(degree, train_data, train_result, test_data, test_result):

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

    print(find_nlargest_index(reg.coef_, feature_name, 10))


if __name__ == '__main__':
    train_data = create_data(1000, 5)
    test_data = create_data(50, 5)

    # print('Griewank function:')
    #
    # train_result_Griewank = create_result(train_data, benchmark.Griewank)
    # test_result_Griewank = create_result(test_data, benchmark.Griewank)
    #
    # Regression(2, train_data, train_result_Griewank, test_data, test_result_Griewank)
    # Regression(3, train_data, train_result_Griewank, test_data, test_result_Griewank)
    # Regression(4, train_data, train_result_Griewank, test_data, test_result_Griewank)

    # train_data_np = np.array(train_data)
    # ax = plt.subplot('Griewank', projection='3d')
    # ax.scatter(train_data_np[:, 0], train_data_np[:, 1], train_result_Griewank, c="r")
    # plt.show()

    print('-----------------------------------------------------')
    print('Schwefel function:')

    train_result_Schwefel = create_result(train_data, benchmark.Schwefel)
    test_result_Schwefel = create_result(test_data, benchmark.Schwefel)

    # Regression(2, train_data, train_result_Schwefel, test_data, test_result_Schwefel)
    Regression(3, train_data, train_result_Schwefel, test_data, test_result_Schwefel)
    Regression(4, train_data, train_result_Schwefel, test_data, test_result_Schwefel)

    # ax = plt.subplot('Schwefel', projection='3d')
    # ax.scatter(train_data_np[:, 0], train_data_np[:, 1], train_result_Schwefel, c="r")
    # plt.show()
    print('-----------------------------------------------------')
    print('Rosenbrock function:')

    train_result_Rosenbrock = create_result(train_data, benchmark.Rosenbrock)
    test_result_Rosenbrock = create_result(test_data, benchmark.Rosenbrock)

    Regression(2, train_data, train_result_Rosenbrock, test_data, test_result_Rosenbrock)
    Regression(3, train_data, train_result_Rosenbrock, test_data, test_result_Rosenbrock)
    Regression(4, train_data, train_result_Rosenbrock, test_data, test_result_Rosenbrock)

    # ax = plt.subplot('Schwefel', projection='3d')
    # ax.scatter(train_data_np[:, 0], train_data_np[:, 1], train_result_Schwefel, c="r")
    # plt.show()
    print('-----------------------------------------------------')
    print('Rastrigin function:')

    train_result_Rastrigin = create_result(train_data, benchmark.Rastrigin)
    test_result_Rastrigin = create_result(test_data, benchmark.Rastrigin)

    Regression(2, train_data, train_result_Rastrigin, test_data, test_result_Rastrigin)
    Regression(3, train_data, train_result_Rastrigin, test_data, test_result_Rastrigin)
    Regression(4, train_data, train_result_Rastrigin, test_data, test_result_Rastrigin)

    print('-----------------------------------------------------')
    print('Ackley function:')

    train_result_Ackley = create_result(train_data, benchmark.Ackley)
    test_result_Ackley = create_result(test_data, benchmark.Ackley)

    Regression(2, train_data, train_result_Ackley, test_data, test_result_Ackley)
    Regression(3, train_data, train_result_Ackley, test_data, test_result_Ackley)
    Regression(4, train_data, train_result_Ackley, test_data, test_result_Ackley)
