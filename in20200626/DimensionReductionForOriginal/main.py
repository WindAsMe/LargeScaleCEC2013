from in20200626.DimensionReductionForOriginal.Sparse import SparseModel
from in20200626.DimensionReductionForOriginal.util import benchmark, help, aim
from in20200626.DimensionReductionForOriginal.DE import DE
import numpy as np
import scipy.stats as stats
import gc


if __name__ == '__main__':
    Func_Dim = 1000
    Model_Dim = 50
    feature_size = 10000
    degree = 2
    func_num = 5
    benchmark_function_summary = benchmark.f_summary(func_num)
    scale_range = [benchmark_function_summary['lower'], benchmark_function_summary['upper']]

    benchmark_function = benchmark.f_evaluation
    complex_evaluate_function = aim.aim_original_function
    name = "benchmark number: " + str(func_num) + " in CEC2013"
    print(name)
    print('scale range: ', scale_range)
    train_data = help.create_data(feature_size, Func_Dim, Model_Dim, scale_range)
    train_label = help.create_result(train_data, benchmark_function, func_num)

    train_data_50 = train_data[:, 0:50].tolist()
    del train_data
    gc.collect()

    reg_Lasso, feature_names = SparseModel.Regression(degree, train_data_50, train_label, feature_size)

    coef, feature_names = help.not_zero_feature(reg_Lasso.coef_[1:], help.feature_names_normalization(feature_names))

    groups = help.group_related_variable(feature_names)
    groups_modified = help.group_modified(groups, feature_names)
    simple_problems_Data_index = help.extract(groups_modified)
    print('Active group: ', len(simple_problems_Data_index), simple_problems_Data_index)
    temp = []
    for group in groups_modified:
        temp += group
    rest_group = []
    temp = list(set(temp))
    for i in range(0, Model_Dim):
        if i not in temp:
            rest_group.append([i])
    if rest_group:
        simple_problems_Data_index += rest_group
    print('extract: ', simple_problems_Data_index)
    print('extract len: ', len(simple_problems_Data_index))
    # groups_Lasso = simple_problems_Data_index
    #
    # """The next is DE optimization"""
    # # Why in first generation has the gap?
    # # Because the grouping strategy firstly do the best features combination in initial population
    # simple_population_size = 20
    # complex_population_size = 500
    # MAX_iteration = 1000
    # # Because for each group, optimization is with intercept, so when combination, we need to minus.
    # intercept = benchmark_function([0] * Dim)
    # print('intercept: ', intercept)
    #
    # scale_range = [-10, 10]
    # max_or_min = 1
    # simple_evaluate_function = aim.fitness_evaluation
    #
    # # simple_init_population = help.init_DE_Population(simple_population_size, Dim, scale_range)
    # # complex_init_population = copy.deepcopy(simple_init_population)
    # # help.write_initial_population('initial_population', name, simple_init_population)
    # # print(init_population)
    # index = [0] * Dim
    # best_simple = 0
    # test_times = 50
    #
    # best_simple_obj_traces = []
    # best_simple_var_traces = []
    # best_complex_obj_traces = []
    #
    # best_simple_obj_traces_average = []
    # best_complex_obj_traces_average = []
    # best_simple_indexes_average = []
    #
    # for t in range(test_times):
    #     print('round ', t)
    #     best_simple_trace, best_simple_index_trace = DE.SimpleProblemsOptimization(Dim, simple_population_size, MAX_iteration,
    #                                                                          benchmark_function,
    #                                                                          simple_evaluate_function, groups_Lasso,
    #                                                                          intercept, max_or_min)
    #     best_simple_var_traces.append(best_simple_index_trace)
    #     real_label = help.create_result(best_simple_index_trace, benchmark_function)
    #     best_simple_obj_traces.append(real_label)
    #     best_complex_trace, best_complex_index = DE.ComplexProblemsOptimization(Dim, complex_population_size,
    #                                                                             MAX_iteration, benchmark_function,
    #                                                                             complex_evaluate_function, max_or_min)
    #     # print('best simple trace: ', best_simple_trace)
    #     # print('best complex trace: ', best_complex_trace)
    #     best_complex_obj_traces.append(best_complex_trace)
    #
    #     help.write_trace("trace_data ", name, t, real_label, best_complex_trace)
    #
    # best_simple_obj_traces = np.array(best_simple_obj_traces)
    # best_complex_obj_traces = np.array(best_complex_obj_traces)
    # best_simple_var_traces = np.array(best_simple_var_traces)
    # print('best simple index: ', best_simple_var_traces)
    # for i in range(len(best_simple_obj_traces[0])):
    #     best_simple_obj_traces_average.append(sum(best_simple_obj_traces[:, i]) / test_times)
    #
    # for i in range(len(best_complex_obj_traces[0])):
    #     best_complex_obj_traces_average.append(sum(best_complex_obj_traces[:, i]) / test_times)
    #
    # for i in range(len(best_simple_var_traces[0])):
    #     best_simple_indexes_average.append(sum(best_simple_var_traces[:, i]) / test_times)
    #
    # x = np.linspace(1, MAX_iteration + 1, MAX_iteration, endpoint=False)
    # help.draw_convergence(x, best_simple_obj_traces_average, best_complex_obj_traces_average, name)
    #
    # help.write_draw('draw_data', name, best_simple_obj_traces, best_complex_obj_traces_average)
    # index_error = []
    # for i in range(len(already_known_global_optima)):
    #     index_error.append(np.abs(best_simple_indexes_average[len(best_simple_indexes_average)-1][i] - already_known_global_optima[i]))
    # help.draw_error(already_known_global_optima, best_simple_indexes_average[len(best_simple_indexes_average)-1], name)
    #
    # generations = [10, 50, 100, 200, 500, 1000]
    # for generation in generations:
    #     print('Mann-whitney statistic for ' + str(generation) + ' th: ',
    #           stats.mannwhitneyu(best_simple_obj_traces[:, generation-1], best_complex_obj_traces[:, generation-1]))

