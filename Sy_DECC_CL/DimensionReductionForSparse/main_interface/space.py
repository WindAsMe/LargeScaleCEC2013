from Sy_DECC_CL.DimensionReductionForSparse.DE import DE
from Sy_DECC_CL.DimensionReductionForSparse.util import help
from Sy_DECC_CL.DimensionReductionForSparse.main_interface.grouping_main import CCDE
from cec2013lsgo.cec2013 import Benchmark
import os.path as path


if __name__ == '__main__':
    # func_num = 11
    Dim = 1000
    NIND = 30
    bench = Benchmark()
    EFs = 3000000
    for func_num in range(2, 16):
        test_time = 1
        name = 'f' + str(func_num)
        benchmark_summary = bench.get_info(func_num)
        scale_range = [benchmark_summary['lower'], benchmark_summary['upper']]
        function = bench.get_function(func_num)

        groups_One = CCDE(Dim)
        for iteration in range(1, 2):

            EFs = 3000000
            name = 'f' + str(func_num)

            """The next is DE optimization"""
            best_indexes, best_obj_trace_CC, up, down = DE.DECC_CL_CCDE(Dim, NIND, iteration, function, scale_range, groups_One)
            delta = []
            for i in range(len(up)):
                delta.append((up[i]-down[i])/(scale_range[1]-scale_range[0]))
            up_bound = max(delta)
            down_bound = min(delta)
            mean = sum(delta) / len(delta)

            data = str(iteration) + ', ' + str(up_bound) + ', ' + str(down_bound) + ', ' + str(mean)
            this_path = path.realpath(__file__)
            data_path = path.dirname(path.dirname(this_path)) + '\\data\\trace\\space\\' + name + '\\space'

            with open(data_path, 'a') as f:
                f.write('[' + data + '],')
                f.write('\n')
                f.close()

        print('    Finished: ', 'function: ', func_num,)


