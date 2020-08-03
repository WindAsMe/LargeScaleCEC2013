from in20200731.DimensionReductionForSparse.Sparse import SparseModel
from in20200731.DimensionReductionForSparse.util import benchmark, help, aim
from in20200731.DimensionReductionForSparse.DE import DE
from sklearn.preprocessing import PolynomialFeatures
from in20200710.BatchSparseTrainOptimization.util import help as new_help
import numpy as np
import time


if __name__ == '__main__':
    Dim = 50
    feature_size = 50000
    degree = 2
    benchmark_function = benchmark.Rastrigin
    mini_batch_size = 1000
    evaluate_function = aim.fitness_evaluation
    scale_range = [-5, 5]
    name = 'Rastrigin'
    time_Lasso_start = time.process_time()
    reg_Lasso, feature_names = SparseModel.Regression(degree, feature_size, Dim, mini_batch_size, scale_range,
                                                      benchmark_function)
    time_Lasso_end = time.process_time()

    coef, feature_names = help.not_zero_feature(reg_Lasso.coef_, help.feature_names_normalization(feature_names))
    time_grouping_start = time.process_time()
    groups_Lasso = new_help.group_DFS(Dim, feature_names, 5)
    groups_random = help.groups_random_create(Dim, 25, 10)
    groups_one = help.groups_one_create(Dim)
    print('Lasso grouping: ', groups_Lasso)
    print('random grouping: ', groups_random)
    #
    # simple_problems_Dim, simple_problems_Data_index = help.extract(groups_modified)
    """The next is DE optimization"""
    # Why in first generation has the gap?
    # Because the grouping strategy firstly do the best features combination in initial population
    simple_population_size = 20
    complex_population_size = 1000
    simple_MAX_iteration = 1000
    complex_MAX_iteration = 20000

    test_times = 2
    efficient_Lasso_iteration_times = 0
    efficient_random_iteration_times = 0
    efficient_one_iteration_times = 0
    efficient_complex_iteration_times = 0

    max_or_min = 1
    poly = PolynomialFeatures(degree=2)

    # print(init_population)
    index = [0] * Dim
    best_simple = 0

    simple_Lasso_problems_trace = []
    simple_random_problems_trace = []
    simple_one_problems_trace = []
    complex_problems_trace = []

    best_Lasso_index_trace = []
    best_random_index_trace = []
    best_one_index_trace = []

    simple_Lasso_problems_trace_average = []
    simple_random_problems_trace_average = []
    simple_one_problems_trace_average = []
    complex_problems_trace_average = []

    best_Lasso_index_average = []
    best_random_index_average = []
    best_one_index_average = []

    time_Lasso_group = 0
    time_random_group = 0
    time_one_group = 0
    time_normal = 0
    for t in range(test_times):
        print('round ', t + 1)
        time1 = time.process_time()
        best_Lasso_obj_trace, best_Lasso_index, e_Lasso_time = DE.SimpleProblemsOptimization(Dim, simple_population_size,
                                                                                  simple_MAX_iteration,
                                                                                  benchmark_function, scale_range,
                                                                                  evaluate_function, groups_Lasso,
                                                                                  max_or_min)
        time2 = time.process_time()
        efficient_Lasso_iteration_times += e_Lasso_time
        simple_Lasso_problems_trace.append(best_Lasso_obj_trace)

        best_Lasso_index_trace.append(best_Lasso_index)
        print('Grouping Lasso optimization time: ', time2 - time1)

        time_Lasso_group += time2 - time1

        best_random_obj_trace, best_random_index, e_random_time = DE.SimpleProblemsOptimization(Dim, simple_population_size,
                                                                                  simple_MAX_iteration,
                                                                                  benchmark_function, scale_range,
                                                                                  evaluate_function, groups_random,
                                                                                  max_or_min)

        time3 = time.process_time()
        efficient_random_iteration_times += e_random_time
        simple_random_problems_trace.append(best_random_obj_trace)
        best_random_index_trace.append(best_random_index)
        print('Grouping random optimization time: ', time3 - time2)
        time_random_group += time3 - time2

        best_one_obj_trace, best_one_index, e_one_time = DE.SimpleProblemsOptimization(Dim, simple_population_size,
                                                                                  simple_MAX_iteration,
                                                                                  benchmark_function, scale_range,
                                                                                  evaluate_function, groups_one,
                                                                                  max_or_min)

        time4 = time.process_time()
        efficient_one_iteration_times += e_one_time
        simple_one_problems_trace.append(best_one_obj_trace)
        best_one_index_trace.append(best_one_index)
        print('Grouping one optimization time: ', time4 - time3)
        time_one_group += time4 - time3

        best_complex_trace, e_complex_time = DE.ComplexProblemsOptimization(Dim, complex_population_size, complex_MAX_iteration,
                                                            evaluate_function, benchmark_function, scale_range,
                                                            max_or_min)
        time5 = time.process_time()
        efficient_complex_iteration_times += e_complex_time
        complex_problems_trace.append(best_complex_trace)
        print('Normal optimization time: ', time5 - time4)
        time_normal += time5 - time4

    print('--------------------------------------------------------------------')
    print('Average Lasso group time: ', time_Lasso_group / test_times)
    print('Average random group time: ', time_random_group / test_times)
    print('Average one group time: ', time_one_group / test_times)
    print('Average normal time: ', time_normal / test_times)
    print('')
    print('Average efficient Lasso group : ', efficient_Lasso_iteration_times / test_times)
    print('Average efficient random group time: ', efficient_random_iteration_times / test_times)
    print('Average efficient one group time: ', efficient_one_iteration_times / test_times)
    print('Average efficient normal time: ', efficient_complex_iteration_times / test_times)
    simple_Lasso_problems_trace = np.array(simple_Lasso_problems_trace)
    simple_random_problems_trace = np.array(simple_random_problems_trace)
    simple_one_problems_trace = np.array(simple_one_problems_trace)

    best_Lasso_index_trace = np.array(best_Lasso_index_trace)
    best_random_index_trace = np.array(best_random_index_trace)
    best_one_index_trace = np.array(best_one_index_trace)

    complex_problems_trace = np.array(complex_problems_trace)

    for i in range(len(simple_Lasso_problems_trace[0])):
        simple_Lasso_problems_trace_average.append(sum(simple_Lasso_problems_trace[:, i]) / test_times)
    for i in range(len(simple_random_problems_trace[0])):
        simple_random_problems_trace_average.append(sum(simple_random_problems_trace[:, i]) / test_times)
    for i in range(len(simple_one_problems_trace[0])):
        simple_one_problems_trace_average.append(sum(simple_one_problems_trace[:, i]) / test_times)

    for i in range(len(complex_problems_trace[0])):
        complex_problems_trace_average.append(sum(complex_problems_trace[:, i]) / test_times)

    for i in range(len(best_Lasso_index_trace[0])):
        best_Lasso_index_average.append(sum(best_Lasso_index_trace[:, i]) / test_times)
    for i in range(len(best_random_index_trace[0])):
        best_random_index_average.append(sum(best_random_index_trace[:, i]) / test_times)
    for i in range(len(best_one_index_trace[0])):
        best_one_index_average.append(sum(best_one_index_trace[:, i]) / test_times)

    help.write_trace(name + '_LASSO', simple_Lasso_problems_trace, simple_Lasso_problems_trace_average)
    help.write_trace(name + '_random', simple_random_problems_trace, simple_random_problems_trace_average)
    help.write_trace(name + '_one', simple_one_problems_trace, simple_one_problems_trace_average)
    help.write_trace(name + '_normal', complex_problems_trace, complex_problems_trace_average)

    x1 = np.linspace(len(groups_Lasso), len(groups_Lasso) * (simple_MAX_iteration + 1), simple_MAX_iteration, endpoint=False)
    x2 = np.linspace(len(groups_random), len(groups_random) * (simple_MAX_iteration + 1), simple_MAX_iteration, endpoint=False)
    x3 = np.linspace(len(groups_one), len(groups_one) * (simple_MAX_iteration + 1), simple_MAX_iteration, endpoint=False)
    x4 = np.linspace(1, complex_MAX_iteration + 1, complex_MAX_iteration, endpoint=False)

    help.draw_obj(x1, x2, x3, x4, simple_Lasso_problems_trace_average, simple_random_problems_trace_average,
                  simple_one_problems_trace_average, complex_problems_trace_average, name)

    x = np.linspace(1, Dim + 1, Dim, endpoint=False)
    help.draw_var(x, best_Lasso_index_average, best_random_index_average, best_one_index_average, index, name)


