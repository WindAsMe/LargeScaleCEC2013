from before20200507.PureCombinationRegression import GAOptimization as optimization
import scipy.stats as stats


def repeatOptimization(function_original, function_Reg, name, reg, dimension, time=10):
    pvalues_max = []
    pvalues_min = []
    for i in range(time):
        min_individual_original, min_index_original = optimization.OptimizationForOriginal(function_original, name +
                                                                                           ' function originally',
                                                                                           1, dimension)
        max_individual_original, max_index_original = optimization.OptimizationForOriginal(function_original, name +
                                                                                           ' function originally',
                                                                                           -1, dimension)
        min_individual_Reg, min_index_Reg = optimization.OptimizationForRegression(function_Reg, name +
                                                                                   ' function for regression',
                                                                                   reg, 1, dimension)
        max_individual_Reg, max_index_Reg = optimization.OptimizationForRegression(function_Reg, name +
                                                                                   ' function for regression',
                                                                                   reg, -1, dimension)
        statistics_max = stats.chisquare(max_index_Reg, f_exp=max_index_original)
        statistics_min = stats.chisquare(min_index_Reg, f_exp=min_index_original)
        print("Chi-square test: ", stats.chi2.isf(0.05, 999), ": ",
              stats.chisquare(max_index_Reg, f_exp=max_index_original))
        print("Chi-square test: ", stats.chi2.isf(0.05, 999), ": ",
              stats.chisquare(min_index_Reg, f_exp=min_index_original))
        pvalues_max.append(statistics_max[1])
        pvalues_min.append(statistics_min[1])
    return pvalues_max, pvalues_min
