from in20200621.DimensionRedunctionForSparse.Sparse import SparseModel
from in20200621.DimensionRedunctionForSparse.util import benchmark, help, aim
from in20200621.DimensionRedunctionForSparse.DE import DE
from sklearn.preprocessing import PolynomialFeatures
from in20200710.BatchSparseTrainOptimization.util import help as new_help
import numpy as np
import time


if __name__ == '__main__':
    Dim = 50
    feature_size = 10000
    degree = 2
    benchmark_function = benchmark.Ackley
    mini_batch_size = 1000
    evaluate_function = aim.fitness_evaluation
    scale_range = [-32, 32]

    time_Lasso_start = time.process_time()
    reg_Lasso, feature_names = SparseModel.Regression(degree, feature_size, Dim, mini_batch_size, scale_range,
                                                      benchmark_function)
    time_Lasso_end = time.process_time()

    coef, feature_names = help.not_zero_feature(reg_Lasso.coef_, help.feature_names_normalization(feature_names))
    time_grouping_start = time.process_time()
    groups_Lasso = new_help.group_DFS(Dim, feature_names, 5)
    groups_random = help.groups_random_create(Dim, 25, 10)
    groups_one = help.groups_one_create(Dim)

    #
    # simple_problems_Dim, simple_problems_Data_index = help.extract(groups_modified)
    """The next is DE optimization"""
    # Why in first generation has the gap?
    # Because the grouping strategy firstly do the best features combination in initial population
    simple_population_size = 30
    complex_population_size = 1500
    simple_MAX_iteration = 1000
    complex_MAX_iteration = simple_MAX_iteration

    max_or_min = 1
    poly = PolynomialFeatures(degree=2)
    simple_init_population = help.init_DE_Population(simple_population_size, Dim, scale_range)
    complex_init_population = help.init_DE_Population(complex_population_size, Dim, scale_range)

    # print(init_population)
    index = [0] * Dim
    best_simple = 0
    test_times = 50

    simple_problems_trace = []
    complex_problems_trace = []
    best_index_trace = []

    simple_problems_trace_average = []
    complex_problems_trace_average = []
    best_index_average = []
    time_group = 0
    time_normal = 0
    for t in range(test_times):
        print('round ', t + 1)
        time1 = time.process_time()
        best_simple_trace, best_index = DE.SimpleProblemsOptimization(Dim, simple_population_size, simple_MAX_iteration,
                                                                      benchmark_function, simple_init_population,
                                                                      evaluate_function, groups, feature_names,
                                                                      coef, max_or_min)
        best_index_trace.append(best_index)
        time2 = time.process_time()
        print('Grouping optimization time: ', time2 - time1)
        time_group += time2 - time1
        best_complex_trace = DE.ComplexProblemsOptimization(Dim, complex_population_size, complex_MAX_iteration,
                                                            complex_init_population, evaluate_function,
                                                            benchmark_function, max_or_min)
        time3 = time.process_time()
        print('normal time: ', time3 - time2)
        time_normal += time3 - time2

        simple_problems_trace.append(best_simple_trace)
        complex_problems_trace.append(best_complex_trace)

    print('Average group time: ', time_group / test_times)
    print('Average normal time: ', time_normal / test_times)
    simple_problems_trace = np.array(simple_problems_trace)
    best_index_trace = np.array(best_index_trace)
    complex_problems_trace = np.array(complex_problems_trace)

    for i in range(len(simple_problems_trace[0])):
        simple_problems_trace_average.append(sum(simple_problems_trace[:, i]) / test_times)

    for i in range(len(complex_problems_trace[0])):
        complex_problems_trace_average.append(sum(complex_problems_trace[:, i]) / test_times)

    for i in range(len(best_index_trace[0])):
        best_index_average.append(sum(best_index_trace[:, i]) / test_times)
    x1 = np.linspace(1, simple_MAX_iteration + 1, simple_MAX_iteration, endpoint=False)
    x2 = np.linspace(1, complex_MAX_iteration + 1, complex_MAX_iteration, endpoint=False)
    help.draw_obj(x1, x2, simple_problems_trace_average, complex_problems_trace_average)

    x = np.linspace(1, Dim + 1, Dim, endpoint=False)
    help.draw_var(x, best_index_average, index)


