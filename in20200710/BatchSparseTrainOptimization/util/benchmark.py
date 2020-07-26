from cec2013lsgo.cec2013 import Benchmark
import numpy as np


bench = Benchmark()

def f_summary(func_num):
    return bench.get_info(func_num)


def f_evaluation(x, func_num):
    function = bench.get_function(func_num)
    return function(np.array(x, dtype='double'))
