from in20200913.DimensionReductionForSparse.util import group
from cec2013lsgo.cec2013 import Benchmark


if __name__ == '__main__':
    Dim = 1000
    size = 100000
    degree = 2
    func_num = 1
    bench = Benchmark()
    function = bench.get_function(func_num)
    benchmark_summary = bench.get_info(func_num)
    scale_range = [benchmark_summary['lower'], benchmark_summary['upper']]
    max_variables_num = 20
    mini_batch_size = 1000
    groups_Lasso = group.group_strategy(func_num, size, Dim, mini_batch_size, scale_range, max_variables_num, function)
