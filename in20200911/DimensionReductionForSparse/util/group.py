from in20200911.DimensionReductionForSparse.util import help
from in20200911.DimensionReductionForSparse.Sparse import SparseModel


def group_strategy(func_num, train_size, Dim, mini_batch_size, scale_range, max_variables_num, function):
    file_name = 'f' + str(func_num)

    reg_Lasso, feature_names = SparseModel.Regression(train_size, Dim, mini_batch_size, scale_range, function)

    coef, feature_names = help.not_zero_feature(reg_Lasso.coef_, help.feature_names_normalization(feature_names))
    print('coef len: ', len(coef))
    groups = help.group_DFS(Dim, feature_names, max_variables_num)
    help.write_grouping(file_name, groups)
    return groups
