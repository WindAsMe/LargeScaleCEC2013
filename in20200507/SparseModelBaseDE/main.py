import numpy as np
import matplotlib.pyplot as plt
from in20200507.SparseModelBaseDE.util import aims, benchmark, SparseTest, help, DE
from in20200507.SparseModelBaseDE.model.MyProblem import MyProblem
import geatpy as ea
import time


def SparseBest(Dimension, function, reg, max_min):
    problem = MyProblem(Dimension, function, reg, max_min)  # 实例化问题对象

    """==============================种群设置==========================="""
    Encoding = 'RI'  # 编码方式
    NIND = 50  # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
    population = ea.Population(Encoding, Field, NIND)

    """===========================算法参数设置=========================="""
    # population.initChrom(NIND)
    # print(population.Chrom)
    myAlgorithm = ea.soea_DE_best_1_L_templet(problem, population)
    myAlgorithm.MAXGEN = Dimension * 10
    myAlgorithm.mutOper.F = 0.5
    myAlgorithm.recOper.XOVR = 0.5
    myAlgorithm.drawing = 0
    """=====================调用算法模板进行种群进化====================="""
    [population, obj_trace, var_trace] = myAlgorithm.run()
    best_gen = np.argmax(obj_trace[:, 1])
    best_ObjV = obj_trace[best_gen, 1]
    # print('最优的目标函数值为：%s' % (best_ObjV))
    # print('最优的决策变量值为：')
    best_index = []
    for i in range(var_trace.shape[1]):
        best_index.append(var_trace[best_gen, i])
    # print(best_index)
    # print('有效进化代数：%s' % (obj_trace.shape[0]))
    # print('最优的一代是第 %s 代' % (best_gen + 1))
    # print('评价次数：%s' % (myAlgorithm.evalsNum))
    # print('时间已过 %s 秒' % (myAlgorithm.passTime))

    return best_index, best_ObjV


def draw(y1, y2, y3):
    plt.tight_layout()
    x1 = np.linspace(100, len(y1) * 100 + 100, len(y1), endpoint=False)
    x2 = np.linspace(101, len(y2) * 100 + 101, len(y2), endpoint=False)
    x3 = np.linspace(221, len(y3) * 100 + 221, len(y3), endpoint=False)
    plt.plot(x1, y1, label='DE + 0 elite(normal DE)')
    plt.plot(x2, y2, label='DE + 1 elite')
    plt.plot(x3, y3, label='DE + 30 nearby elite')
    plt.title('Assist Generation')
    plt.xlabel('The number of evaluation')
    plt.ylabel('fitness')

    plt.legend()
    plt.show()


def draw_average(matrix):
    result = []
    m = np.array(matrix)
    for i in range(0, len(m[0])):
        result.append(sum(m[:, i]) / len(m))
    return result


if __name__ == '__main__':
    Dimension = 50
    for_min = 1
    test_times = 50
    function_name = 'Schwefel'
    aim_original = aims.aim_Schwefel_original
    aim_two = aims.aim_Schwefel_two
    benchmark_function = benchmark.Schwefel
    bias_original = 121
    bias_one = 120
    # Schwefel: min
    # Rosenbrock: min
    # Rastrigin: min
    # Ackley: min

    reg = SparseTest.SparseModeling(Dimension, benchmark_function, 2)

    # As prior knowledge
    best_index_one = help.find_best_worst(aim_two, for_min, Dimension, reg)
    best_indexes_near = help.create_points(benchmark_function, best_index_one[0], number=30)

    original_1 = []
    assist_one_1 = []
    assist_near_1 = []

    original_5 = []
    assist_one_5 = []
    assist_near_5 = []

    original_10 = []
    assist_one_10 = []
    assist_near_10 = []

    original_20 = []
    assist_one_20 = []
    assist_near_20 = []

    original_30 = []
    assist_one_30 = []
    assist_near_30 = []

    original_50 = []
    assist_one_50 = []
    assist_near_50 = []

    original_100 = []
    assist_one_100 = []
    assist_near_100 = []

    original_200 = []
    assist_one_200 = []
    assist_near_200 = []

    original_500 = []
    assist_one_500 = []
    assist_near_500 = []

    original_1000 = []
    assist_one_1000 = []
    assist_near_1000 = []

    draw_assist_one_50 = []
    draw_original_50 = []
    draw_assist_near_50 = []

    draw_assist_one_100 = []
    draw_original_100 = []
    draw_assist_near_100 = []

    draw_assist_one_200 = []
    draw_original_200 = []
    draw_assist_near_200 = []

    draw_assist_one_500 = []
    draw_original_500 = []
    draw_assist_near_500 = []

    draw_assist_one_1000 = []
    draw_original_1000 = []
    draw_assist_near_1000 = []

    time_DE = 0
    time_DE_1 = 0
    time_DE_10 = 0
    f = open('D:\CS2019KYUTAI\PythonProject\SparseModeling\in20200507\SparseModelBaseDE\data\\raw_data', 'a')
    f.write(function_name + ': ' + str(Dimension))
    f.close()
    for i in range(test_times):
        print('round ', i + 1)

        # find_best_worst(iteration, for_max, reg, Dimension)
        time1 = time.time()
        original_index, original_trace, init_Chrom = DE.NoAssistOptimization(Dimension, aim_original, reg, for_min)
        time2 = time.time()
        assist_one_best_index, assist_one_trace = DE.AssistOptimization(Dimension, aim_original, reg, for_min,
                                                                     best_index_one, init_Chrom)
        time3 = time.time()
        assist_near_best_index, assist_near_trace = DE.AssistOptimization(Dimension, aim_original, reg, for_min,
                                                                     best_indexes_near, init_Chrom)
        time4 = time.time()
        help.write_raw_data('raw_data', i, original_trace[:, 1], assist_one_trace[:, 1], assist_near_trace[:, 1])
        time_DE += time2 - time1
        time_DE_1 += time3 - time2
        time_DE_10 += time4 - time3
        print('DE time: ', time2 - time1, 'DE + 1 elite time: ', time3 - time2, 'DE + 10 elite time: ', time4 - time3,)
        original_1.append(original_trace[:, 1][0+bias_original])
        original_5.append(original_trace[:, 1][4+bias_original])
        original_10.append(original_trace[:, 1][9+bias_original])
        original_20.append(original_trace[:, 1][19+bias_original])
        original_30.append(original_trace[:, 1][29+bias_original])

        assist_one_1.append(assist_one_trace[:, 1][0+bias_one])
        assist_one_5.append(assist_one_trace[:, 1][4+bias_one])
        assist_one_10.append(assist_one_trace[:, 1][9+bias_one])
        assist_one_20.append(assist_one_trace[:, 1][19+bias_one])
        assist_one_30.append(assist_one_trace[:, 1][29+bias_one])

        assist_near_1.append(assist_near_trace[:, 1][0])
        assist_near_5.append(assist_near_trace[:, 1][4])
        assist_near_10.append(assist_near_trace[:, 1][9])
        assist_near_20.append(assist_near_trace[:, 1][19])
        assist_near_30.append(assist_near_trace[:, 1][29])

        original_50.append(original_trace[:, 1][49+bias_original])
        assist_one_50.append(assist_one_trace[:, 1][49+bias_one])
        assist_near_50.append(assist_near_trace[:, 1][49])

        draw_original_50.append(original_trace[:, 1][:50])
        draw_assist_one_50.append(assist_one_trace[:, 1][:50])
        draw_assist_near_50.append(assist_near_trace[:, 1][:50])

        original_100.append(original_trace[:, 1][99+bias_original])
        assist_one_100.append(assist_one_trace[:, 1][99+bias_one])
        assist_near_100.append(assist_near_trace[:, 1][99])

        draw_original_100.append(original_trace[:, 1][:100])
        draw_assist_one_100.append(assist_one_trace[:, 1][:100])
        draw_assist_near_100.append(assist_near_trace[:, 1][:100])

        original_200.append(original_trace[:, 1][199+bias_original])
        assist_one_200.append(assist_one_trace[:, 1][199+bias_one])
        assist_near_200.append(assist_near_trace[:, 1][199])

        draw_original_200.append(original_trace[:, 1][:200])
        draw_assist_one_200.append(assist_one_trace[:, 1][:200])
        draw_assist_near_200.append(assist_near_trace[:, 1][:200])

        original_500.append(original_trace[:, 1][499+bias_original])
        assist_one_500.append(assist_one_trace[:, 1][499+bias_one])
        assist_near_500.append(assist_near_trace[:, 1][499])

        draw_original_500.append(original_trace[:, 1][:500])
        draw_assist_one_500.append(assist_one_trace[:, 1][:500])
        draw_assist_near_500.append(assist_near_trace[:, 1][:500])

        original_1000.append(original_trace[:, 1][999+bias_original])
        assist_one_1000.append(assist_one_trace[:, 1][999+bias_one])
        assist_near_1000.append(assist_near_trace[:, 1][999])

        draw_original_1000.append(original_trace[:, 1][:1000])
        draw_assist_one_1000.append(assist_one_trace[:, 1][:1000])
        draw_assist_near_1000.append(assist_near_trace[:, 1][:1000])
    print('average time: ', time_DE / test_times, time_DE_1 / test_times, time_DE_10 / test_times)
    draw_assist_one_50 = draw_average(draw_assist_one_50)
    draw_original_50 = draw_average(draw_original_50)
    draw_assist_near_50 = draw_average(draw_assist_near_50)

    draw_assist_one_100 = draw_average(draw_assist_one_100)
    draw_original_100 = draw_average(draw_original_100)
    draw_assist_near_100 = draw_average(draw_assist_near_100)

    draw_assist_one_200 = draw_average(draw_assist_one_200)
    draw_original_200 = draw_average(draw_original_200)
    draw_assist_near_200 = draw_average(draw_assist_near_200)

    draw_assist_one_500 = draw_average(draw_assist_one_500)
    draw_original_500 = draw_average(draw_original_500)
    draw_assist_near_500 = draw_average(draw_assist_near_500)

    draw_assist_one_1000 = draw_average(draw_assist_one_1000)
    draw_original_1000 = draw_average(draw_original_1000)
    draw_assist_near_1000 = draw_average(draw_assist_near_1000)

    p_f_1, ave_o_1, ave_one_a_1, ave_near_a_1 = help.statistic(original_1, assist_one_1, assist_near_1)
    p_f_5, ave_o_5, ave_one_a_5, ave_near_a_5 = help.statistic(original_5, assist_one_5, assist_near_5)
    p_f_10, ave_o_10, ave_one_a_10, ave_near_a_10 = help.statistic(original_10, assist_one_10,  assist_near_10)
    p_f_20, ave_o_20, ave_one_a_20, ave_near_a_20 = help.statistic(original_20, assist_one_20, assist_near_20)
    p_f_30, ave_o_30, ave_one_a_30, ave_near_a_30 = help.statistic(original_30, assist_one_30, assist_near_30)
    p_f_50, ave_o_50, ave_one_a_50, ave_near_a_50 = help.statistic(original_50, assist_one_50, assist_near_50)
    p_f_100, ave_o_100, ave_one_a_100, ave_near_a_100 = help.statistic(original_100, assist_one_100, assist_near_100)
    p_f_200, ave_o_200, ave_one_a_200, ave_near_a_200 = help.statistic(original_200, assist_one_200, assist_near_200)
    p_f_500, ave_o_500, ave_one_a_500, ave_near_a_500 = help.statistic(original_500, assist_one_500, assist_near_500)
    p_f_1000, ave_o_1000, ave_one_a_1000, ave_near_a_1000 = help.statistic(original_1000, assist_one_1000,
                                                                           assist_near_1000)

    f = open('D:\CS2019KYUTAI\PythonProject\SparseModeling\in20200507\SparseModelBaseDE\data\data_statistic', 'a')
    f.write('At ' + str(Dimension) + 'D: ' + function_name + '\n')
    f.close()
    help.write_statistic('data_statistic', '1', str(p_f_1), original_1, assist_one_1, assist_near_1)
    help.write_statistic('data_statistic', '5', str(p_f_5), original_5, assist_one_5, assist_near_5)
    help.write_statistic('data_statistic', '10', str(p_f_10), original_10, assist_one_10, assist_near_10)
    help.write_statistic('data_statistic', '20', str(p_f_20), original_20, assist_one_20, assist_near_20)
    help.write_statistic('data_statistic', '30', str(p_f_30), original_30, assist_one_30, assist_near_30)
    help.write_statistic('data_statistic', '50', str(p_f_50), original_50, assist_one_50, assist_near_50)
    help.write_statistic('data_statistic', '100', str(p_f_100), original_100, assist_one_100, assist_near_100)
    help.write_statistic('data_statistic', '200', str(p_f_200), original_200, assist_one_200, assist_near_200)
    help.write_statistic('data_statistic', '500', str(p_f_500), original_500, assist_one_500, assist_near_500)
    help.write_statistic('data_statistic', '1000', str(p_f_1000), original_1000, assist_one_1000, assist_near_1000)
    f = open('D:\CS2019KYUTAI\PythonProject\SparseModeling\in20200507\SparseModelBaseDE\data\data_statistic', 'a')
    f.write('\n')
    f.close()

    f = open('D:\CS2019KYUTAI\PythonProject\SparseModeling\in20200507\SparseModelBaseDE\data\data_draw', 'a')
    f.write('At ' + str(Dimension) + 'D: ' + function_name + '\n')
    f.close()
    help.write_draw('data_draw', draw_original_50, draw_assist_one_50, draw_assist_near_50)
    help.write_draw('data_draw', draw_original_100, draw_assist_one_100, draw_assist_near_100)
    help.write_draw('data_draw', draw_original_200, draw_assist_one_200, draw_assist_near_200)
    help.write_draw('data_draw', draw_original_500, draw_assist_one_500, draw_assist_near_500)
    help.write_draw('data_draw', draw_original_1000, draw_assist_one_1000, draw_assist_near_1000)
    f = open('D:\CS2019KYUTAI\PythonProject\SparseModeling\in20200507\SparseModelBaseDE\data\data_draw', 'a')
    f.write('\n')
    f.close()

    draw(draw_original_50, draw_assist_one_50, draw_assist_near_50)
    draw(draw_original_100, draw_assist_one_100, draw_assist_near_100)
    draw(draw_original_200, draw_assist_one_200, draw_assist_near_200)
    draw(draw_original_500, draw_assist_one_500, draw_assist_near_500)
    draw(draw_original_1000, draw_assist_one_1000, draw_assist_near_1000)

    print('p_value in 1: ', p_f_1, 'average No-assisted: ', ave_o_1, 'average one Assisted: ', ave_one_a_1,
         'average near Assisted: ', ave_near_a_1)
    print('p_value in 5: ', p_f_5, 'average No-assisted: ', ave_o_5, 'average one Assisted: ', ave_one_a_5,
          'average near Assisted: ', ave_near_a_5)
    print('p_value in 10: ', p_f_10, 'average No-assisted: ', ave_o_10, 'average one Assisted: ', ave_one_a_10,
          'average near Assisted: ', ave_near_a_10)
    print('p_value in 20: ', p_f_20, 'average No-assisted: ', ave_o_20, 'average one Assisted: ', ave_one_a_20,
          'average near Assisted: ', ave_near_a_20)
    print('p_value in 30: ', p_f_30, 'average No-assisted: ', ave_o_30, 'average one Assisted: ', ave_one_a_30,
          'average near Assisted: ', ave_near_a_30)
    print('p_value in 50: ', p_f_50, 'average No-assisted: ', ave_o_50, 'average one Assisted: ', ave_one_a_50,
          'average near Assisted: ', ave_near_a_50)
    print('p_value in 100: ', p_f_100, 'average No-assisted: ', ave_o_100, 'average one Assisted: ', ave_one_a_100,
          'average near Assisted: ', ave_near_a_100)
    print('p_value in 200: ', p_f_200, 'average No-assisted: ', ave_o_200, 'average one Assisted: ', ave_one_a_200,
          'average near Assisted: ', ave_near_a_200)
    print('p_value in 500: ', p_f_500, 'average No-assisted: ', ave_o_500, 'average one Assisted: ', ave_one_a_500,
          'average near Assisted: ', ave_near_a_500)
    print('p_value in 1000: ', p_f_1000, 'average No-assisted: ', ave_o_1000, 'average one Assisted: ', ave_one_a_1000,
          'average near Assisted: ', ave_near_a_1000)















