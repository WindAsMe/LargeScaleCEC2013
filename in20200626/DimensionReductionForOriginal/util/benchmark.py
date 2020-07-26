from cec2013lsgo.cec2013 import Benchmark
import numpy as np


def f_summary(func_num):
    bench = Benchmark()
    return bench.get_info(func_num)


def f_evaluation(x, func_num):
    bench = Benchmark()
    function = bench.get_function(func_num)
    return function(np.array(x, dtype='double'))
