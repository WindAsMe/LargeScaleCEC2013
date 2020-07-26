from in20200706.SparseModelForGroupingOptimization.util import help, benchmark
import numpy as np


def write_train_data(path, Func_num):
    benchmark_function_summary = benchmark.f_summary(Func_num)
    scale_range = [benchmark_function_summary['lower'], benchmark_function_summary['upper']]
    train_data = help.create_data(feature_size, Func_Dim, Model_Dim, scale_range)
    with open(path + 'train_data\\' + str(Func_num) + '.txt', 'w') as file:
        train_data_temp = train_data[:, 0:50]
        for data in train_data_temp:
            for index in range(len(data)):
                if index == len(data) - 1:
                    file.write(str(data[index]) + '\n')
                else:
                    file.write(str(data[index]) + ' ')
    file.close()
    return train_data


def write_train_label(path, train_data, Func_Num):
    benchmark_function = benchmark.f_evaluation
    train_label = help.create_result(train_data, benchmark_function, Func_num)
    with open(path + 'train_label\\' + str(Func_num) + '.txt', 'w') as file:
        for index in range(len(train_label)):
            if index == len(train_label) - 1:
                file.write(str(train_label[index]))
            else:
                file.write(str(train_label[index]) + '\n')
    file.close()


if __name__ == '__main__':
    path = 'D:\CS2019KYUTAI\PythonProject\SparseModeling\in20200706\SparseModelForGroupingOptimization\data\\'
    Func_Dim = 1000
    Model_Dim = 50
    feature_size = 10000
    Func_num = 15
    benchmark_function = benchmark.f_evaluation

    train_data = write_train_data(path, Func_num)
    write_train_label(path, train_data, Func_num)

    # with open(path + 'train_data_' + str(Func_num) + '.txt', 'r') as file:
    #     train_data = file.readlines()
    #     for i in range(len(train_data)):
    #         train_data[i] = (train_data[i].rstrip('\n')).split(' ')
    #         for j in range(Dim):
    #             if j < Model_Dim:
    #                 train_data[i][j] = float(train_data[i][j])
    #             else:
    #                 train_data[i].append(0)
    # file.close()
    #
    # print('Function num: ', Func_num)
    # train_label = help.create_result(train_data, benchmark_function, Func_num)
    # with open(path + 'train_label_' + str(Func_num) + '.txt', 'w') as file:
    #     for index in range(len(train_label)):
    #         if index == len(train_label) - 1:
    #             file.write(str(train_label[index]))
    #         else:
    #             file.write(str(train_label[index]) + '\n')
    # file.close()


