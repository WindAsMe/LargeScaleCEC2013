from in20200808.DimensionReductionForSparse.util import benchmark, group


if __name__ == '__main__':
    Dim = 1000
    feature_size = 100000
    degree = 2
    func_num = 1
    benchmark_summary = benchmark.f_summary(func_num)
    scale_range = [benchmark_summary['lower'], benchmark_summary['upper']]
    max_variables_num = 50
    mini_batch_size = 1000
    groups_Lasso = group.group_strategy(func_num, degree, feature_size, Dim, mini_batch_size, scale_range, max_variables_num)
