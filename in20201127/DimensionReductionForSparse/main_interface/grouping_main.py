from in20201127.DimensionReductionForSparse.util import help
from in20201127.DimensionReductionForSparse.Sparse import SparseModel
from cec2013lsgo.cec2013 import Benchmark


if __name__ == '__main__':
    Dim = 1000
    size = 5000
    degree = 3
    func_num = 5
    bench = Benchmark()
    group_dim = 50
    max_variables_num = 50

    # current index from (i*50, (i+1)*50)
    for func_num in range(4, 5):
        file_name = 'f' + str(func_num)
        function = bench.get_function(func_num)
        benchmark_summary = bench.get_info(func_num)
        scale_range = [benchmark_summary['lower'], benchmark_summary['upper']]

        All_groups = []
        for current_index in range(0, 20):
            Lasso_model, Feature_names = SparseModel.Regression(degree, size, Dim, group_dim, current_index, scale_range, function)

            # Grouping
            coef, Feature_names = help.not_zero_feature(Lasso_model.coef_,
                                                        help.feature_names_normalization(Feature_names))
            groups = help.group_DFS(group_dim, Feature_names, max_variables_num)

            bias = current_index * group_dim
            for g in groups:
                for i in range(len(g)):
                    g[i] += bias

            All_groups.extend(groups)
            # We need to check the relationship between new groups and previous groups
            # for group in All_groups:
        print(All_groups)
