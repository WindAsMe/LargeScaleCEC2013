import numpy as np
from in20200530.SparseModelContourf.util import aims, SparseTest, help, DE, temp_benchmark
from in20200530.SparseModelContourf.util import benchmark
import random


def create_points(number, scale):
    points_index = []
    for i in range(number):
        point = []
        for j in range(scale):
            point.append(random.uniform(-10, 10))
        points_index.append(point)
    return np.array(points_index)


if __name__ == '__main__':
    Dimension = 2
    for_min = 1
    aim_two = aims.aim_Ackley_two
    temp_benchmark_function = temp_benchmark.Ackley
    benchmark_function = benchmark.Ackley

    points_index = create_points(10, 2)
    max_or_min = 1
    reg = SparseTest.SparseModeling(Dimension, benchmark_function, 2)

    best_index = [DE.OptimizationForSparse(Dimension, aim_two, reg, max_or_min)]
    best_indexes = help.create_points(best_index[0], 5, 0.5)

    chrom_1 = help.get_chrom_10(points_index, aim_two, reg, best_index)
    chrom_2 = help.get_chrom_10(points_index, aim_two, reg, best_indexes)

    help.draw(points_index, temp_benchmark_function)
    help.draw_elite_1(best_index[0], chrom_1, temp_benchmark_function)
    help.draw_elite_10(best_index[0], best_indexes, chrom_2, temp_benchmark_function)


















