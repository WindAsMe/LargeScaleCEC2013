from in20200808.DimensionReductionForSparse.util import help
from in20200808.DimensionReductionForSparse.Sparse import SparseModel


def group_strategy(Func_num, degree, train_size, Func_Dim, mini_batch_size, scale_range, max_variables_num):
    file_name = 'f' + str(Func_num)

    reg_Lasso, feature_names = SparseModel.Regression(degree, train_size, Func_Dim, mini_batch_size, scale_range,
                                                      Func_num)

    coef, feature_names = help.not_zero_feature(reg_Lasso.coef_, help.feature_names_normalization(feature_names))
    print('coef len: ', len(coef))
    groups = help.group_DFS(Func_Dim, feature_names, max_variables_num)
    help.write_grouping(file_name, groups)
    return groups
