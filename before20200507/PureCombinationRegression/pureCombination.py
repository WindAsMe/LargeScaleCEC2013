from before20200507.TestBenchmark import benchmark
from sklearn import linear_model
from sklearn.metrics import r2_score
import random
import scipy.stats as stats
from before20200507.PureCombinationRegression import RepeatOptimization as optimization, aims


def min_max_normalization(data):
    result = []
    max_value = max(data)
    min_value = min(data)
    for d in data:
        result.append((d - min_value) / (max_value - min_value))
    return result


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


def pure_combination(features, degree):
    data = []
    for feature in features:
        # print('feature: ', feature)
        temp = []
        for i in range(1, degree + 1):
            for f in feature:
                temp.append(pow(f, i))
        data.append(temp)
    return data


def feature_combination(features_size, degree):
    feature_name = []
    for i in range(0, features_size):
        feature_name.append('x' + str(i))
    for i in range(2, degree + 1):
        for j in range(0, features_size):
            feature_name.append('x' + str(j) + '^' + str(i))
    return feature_name


def Regression(train_data_poly, train_label, test_data_poly, test_label):

    reg = linear_model.Lasso(max_iter=100000)
    reg.fit(train_data_poly, train_label)
    train_predict = reg.predict(train_data_poly)
    test_predict = reg.predict(test_data_poly)

    # The following is the result
    # trainMSE = mean_squared_error(train_result, train_predict)
    # testMSE = mean_squared_error(test_result, test_predict)

    trainR2 = r2_score(train_label, train_predict)
    testR2 = r2_score(test_label, test_predict)
    print("The max coefficient: ", max(reg.coef_))
    print("The min coefficient: ", min(reg.coef_))
    print("The average of coefficients: ", sum(reg.coef_) / len(reg.coef_))
    print("train data R2: ", trainR2, "    test data R2: ", testR2)
    return reg


def statistic_Judgement(real_label, predict_label):
    print('The real label: ', real_label)
    print('The predict label: ', predict_label)
    print('The difference between none-normalization label: ',
          stats.wilcoxon(real_label, predict_label, correction=True))

    print('The similarity between none-normalization label: ',
          stats.pearsonr(real_label, predict_label))


def find_valid_coefficients(feature_name, coef):
    max_value = max(coef)
    valid_feature_name = []
    valid_coef = []
    for i in range(len(coef)):
        if abs(coef[i]) > 0.001 and abs(coef[i]) > max_value * 0.05:
            valid_feature_name.append(feature_name[i])
            valid_coef.append(coef[i])
    print("All coefficients: ", len(coef))
    print("Valid coefficients: ", len(valid_feature_name))


if __name__ == '__main__':
    train_data = create_data(50000, 1000)
    test_data = create_data(5000, 1000)

    train_data_poly = pure_combination(train_data, 4)
    test_data_poly = pure_combination(test_data, 4)

    feature_name = feature_combination(1000, 4)

    train_label = create_result(train_data, benchmark.Griewank)
    test_label = create_result(test_data, benchmark.Griewank)

    reg = Regression(train_data_poly, train_label, test_data_poly, test_label)

    statistic_Judgement(test_label, reg.predict(test_data_poly))
    find_valid_coefficients(feature_name, reg.coef_)

    print("The NEXT is the results for optimization-------------------------------")

    name = 'Griewank'
    function_original = aims.aim_Griewank_original
    function_Reg = aims.aim_Griewank_Reg
    benchmark_function = benchmark.Griewank
    dimension = 1000

    pvalues_max, pvalues_min = optimization.repeatOptimization(function_original, function_Reg, name, reg, dimension, 5)
    print("p_value for best individual: ", pvalues_max)
    print("p_value for worst individual: ", pvalues_min)
    # x = np.linspace(0, 999, 1000)
    # dif_max = []
    # dif_min = []
    # for i in range(len(max_index_Reg)):
    #     dif_max.append(max_index_Reg[i] - max_index_original[i])
    # for i in range(len(min_index_Reg)):
    #     dif_min.append(min_index_Reg[i] - min_index_original[i])
    #
    # fig = plt.figure()
    # ax1 = fig.add_subplot(2, 1, 1)
    # ax2 = fig.add_subplot(2, 1, 2)
    # ax1.plot(x, dif_max)
    # ax2.plot(x, dif_min)
    # plt.show()
