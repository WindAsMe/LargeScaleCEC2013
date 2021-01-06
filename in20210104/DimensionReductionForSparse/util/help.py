import random
import copy
import numpy as np
import matplotlib.pyplot as plt


def create_local_model_data(scale, total_dim, group_dim, current_index, scale_range):
    total_data = np.zeros((scale, total_dim))
    for d in total_data:
        for i in range(current_index*group_dim, (current_index+1)*group_dim):
            d[i] = random.uniform(scale_range[0], scale_range[1])
    real_data = total_data[:, current_index*group_dim:(current_index+1)*group_dim]
    return total_data, real_data


def create_data(scale, dim, scale_range):
    data = []
    for j in range(0, scale):
        temp = []
        for i in range(0, dim):
            temp.append(random.uniform(scale_range[0], scale_range[1]))
        data.append(temp)
    return np.array(data, dtype='double')


def create_result(train_data, f):
    result = []
    for x in train_data:
        result.append(f(x))
    return np.array(result)


# Change the features ['1', '1^2', '1 2^2'] to [[1], [1, 1], [1, 2, 2]]
def feature_names_normalization(feature_names):
    result = []
    for s in feature_names:
        if s.isdigit():
            result.append([int(s)])
            continue
        else:
            temp = []
            s_list = s.split(' ')
            for sub_s in s_list:
                if sub_s.isdigit():
                    temp.append(int(sub_s))
                else:
                    sub_s_list = sub_s.split('^')
                    for i in range(int(sub_s_list[1])):
                        temp.append(int(sub_s_list[0]))
            result.append(temp)
    return result


def not_zero_feature(coef, feature_names):
    new_coef = []
    new_feature_names = []
    for i in range(len(coef)):
        if coef[i] != 0:
            new_coef.append(coef[i])
            new_feature_names.append(feature_names[i])
    return new_coef, new_feature_names


def have_same_element(l1, l2):
    for e in l1:
        if e in l2:
            return True
    return False


def list_combination(l1, l2):
    for l in l2:
        if l not in l1:
            l1.append(l)
    return l1


def group_DFS(Dim, feature_names, max_variable_num):
    temp_feature_names = copy.deepcopy(feature_names)
    groups_element = []
    groups_index = []

    while temp_feature_names:
        elements = temp_feature_names.pop(0)
        group_element = elements
        group_index = [feature_names.index(elements)]
        flag = [1]
        for element in elements:
            help_DFS(group_element, group_index, element, temp_feature_names, feature_names, flag, max_variable_num)
        group_element = list(set(group_element))
        interactions = []
        for name in temp_feature_names:
            interaction = [a for a in group_element if a in name]
            if len(interaction) > 0:
                interactions.append(name)
        for name in interactions:
            temp_feature_names.remove(name)
        groups_element.append(group_element)
        groups_index.append(group_index)

    verify = []
    for group in groups_element:
        verify.extend(group)
    final_g = []
    for i in range(Dim):
        if i not in verify:
            final_g.append(i)
    groups_element.append(final_g)
    return groups_element


def help_DFS(group_element, group_index, element, temp_feature_names, feature_names, flag, max_variable_num):
    if flag[0] >= max_variable_num:
        return
    else:
        i = -1
        while temp_feature_names:
            i += 1
            if i >= len(temp_feature_names):
                return
            else:
                if element in temp_feature_names[i]:
                    temp_elements = temp_feature_names.pop(i)
                    group_element.extend(temp_elements)
                    group_index.append(feature_names.index(temp_elements))

                    flag[0] = len(set(group_element))
                    if flag[0] >= max_variable_num:
                        return
                    for temp_element in temp_elements:
                        help_DFS(group_element, group_index, temp_element, temp_feature_names, feature_names, flag,
                                 max_variable_num)
                        if flag[0] >= max_variable_num:
                            return


def draw_obj(x, y1, label, name):
    plt.plot(x, y1, label=label)
    plt.xlabel('Evaluation times')
    plt.ylabel('Fitness')
    plt.legend()
    plt.savefig(
        'D:\CS2019KYUTAI\PythonProject\SparseModeling\in20200913\DimensionReductionForSparse\data\pic\\' + name + '_obj')
    plt.show()


def write_obj_trace(path, fileName, trace):
    full_path = "D:\CS2019KYUTAI\PythonProject\SparseModeling\in20201230\DimensionReductionForSparse\data\\trace\obj\\" + path + "\\" + fileName
    with open(full_path, 'a') as f:
        f.write('[')
        for i in range(len(trace)):
            if i == len(trace) - 1:
                f.write(str(trace[i]))
            else:
                f.write(str(trace[i]) + ', ')
        f.write('],')
        f.write('\n')
        f.close()


def write_var_trace(path, fileName, trace):
    full_path = "D:\CS2019KYUTAI\PythonProject\SparseModeling\in20200913\DimensionReductionForSparse\data\\final\\var\\" + path + "\\" + fileName
    with open(full_path, 'a') as f:
        f.write('[')
        for i in range(len(trace)):
            f.write('[')
            for j in range(len(trace[i])):
                if j == len(trace[i]) - 1:
                    f.write(str(trace[i][j]) + ', ')
                else:
                    f.write(str(trace[i][j]))
            f.write(']')
        f.write(']')
        f.write('\n')
        f.close()


def write_grouping(path, groups):
    with open('D:\CS2019KYUTAI\PythonProject\SparseModeling\in20201127\DimensionReductionForSparse\grouping\\' + path, 'a') as file:
        file.write('[')
        for g in groups:
            file.write(str(g) + ', ')
        file.write(']')
    file.close()


def write_cost(path, cost):
    with open('D:\CS2019KYUTAI\PythonProject\SparseModeling\data\cost\\' + path, 'a') as file:
        file.write(str(cost) + ',')
    file.close()


def set_Chrom_zero(pop, group):
    for i in range(len(pop.Chrom)):
        for j in range(len(pop.Chrom[i])):
            if j not in group:
                pop.Chrom[i][j] = 0
    pop.Phen = pop.Chrom


def preserve(var_traces, benchmark_function):
    obj_traces = []
    for v in var_traces:
        obj_traces.append(benchmark_function(v))
    for i in range(len(obj_traces) - 1):
        if obj_traces[i] < obj_traces[i + 1]:
            var_traces[i + 1] = var_traces[i]
            obj_traces[i + 1] = obj_traces[i]
    return var_traces, obj_traces


def same_elements_in_population(pop1, pop2):
    ns = 0
    nf = 0
    for chrom in pop1.Chrom:
        if chrom in pop2.Chrom:
            ns += 1
        else:
            nf += 1
    return ns, nf


def draw_summary(x, x_LASSO, x_DECC_DG, x_DECC_D, x_CCDE_LASSO, LASSO_ave, Normal_ave, One_ave,
                      DECC_DG_ave, DECC_D_ave, DECC_G_ave, CCDE_LASSO_ave, name):
    plt.semilogy(x_LASSO, LASSO_ave, label='DECC-L')
    plt.semilogy(x, One_ave, label='CCDE')
    plt.semilogy(x, Normal_ave, label='Normal')
    plt.semilogy(x, DECC_G_ave, label='DECC-G')
    plt.semilogy(x_DECC_D, DECC_D_ave, label='DECC-D')
    plt.semilogy(x_DECC_DG, DECC_DG_ave, label='DECC-DG')
    plt.semilogy(x_CCDE_LASSO, CCDE_LASSO_ave, label='DECC-LC')

    # plt.plot(x_LASSO, LASSO_ave, label='DECC-L')
    # plt.plot(x, One_ave, label='CCDE')
    # plt.plot(x, Normal_ave, label='Normal')
    # plt.plot(x, DECC_G_ave, label='DECC-G Grouping')
    # plt.plot(x_DECC_D, DECC_D_ave, label='DECC-D Grouping')
    # plt.plot(x_DECC_DG, DECC_DG_ave, label='DECC-DG Grouping')
    # plt.plot(x_CCDE_LASSO, CCDE_LASSO_ave, label='DECC-LC')

    plt.xlabel('Evaluation times')
    plt.ylabel('Fitness')
    plt.legend()
    plt.savefig(
        'D:\CS2019KYUTAI\PythonProject\SparseModeling\data\\pic\\' + name + '_obj')
    plt.show()


def write_final(d1, d2, d3, d4):
    with open('D:\CS2019KYUTAI\PythonProject\SparseModeling\in20200913\DimensionReductionForSparse\data\\final\\final\\LASSO', 'a') as file:
        file.write(str(d1) + ', ')
    file.close()
    with open('D:\CS2019KYUTAI\PythonProject\SparseModeling\in20200913\DimensionReductionForSparse\data\\final\\final\\Normal', 'a') as file:
        file.write(str(d2) + ', ')
    file.close()
    with open('D:\CS2019KYUTAI\PythonProject\SparseModeling\in20200913\DimensionReductionForSparse\data\\final\\final\\One', 'a') as file:
        file.write(str(d3) + ', ')
    file.close()
    with open('D:\CS2019KYUTAI\PythonProject\SparseModeling\in20200913\DimensionReductionForSparse\data\\final\\final\\Random', 'a') as file:
        file.write(str(d4) + ', ')
    file.close()


def population_initialization(Dim, scale_range, size):
    population = []
    for s in range(size):
        temp = []
        for i in range(Dim):
            temp.append(random.uniform(scale_range[0], scale_range[1]))
        population.append(temp)
    return np.array(population)


def write_elite(data, func_name):
    with open('D:\CS2019KYUTAI\PythonProject\SparseModeling\in20200913\DimensionReductionForSparse\data\\elite\\' + func_name, 'a') as file:
        file.write('[')
        for i in range(len(data)):
            if i != len(data) - 1:
                file.write(str(data[i]) + ', ')
            else:
                file.write(str(data[i]))
        file.write('],')
        file.write('\n')
    file.close()


# Return: True(separable) False(none-separable)
def Differential(e1, e2, function):
    index_1 = np.zeros((1, 1000))[0]
    index_2 = np.zeros((1, 1000))[0]

    intercept = function(index_1)
    index_1[e1] = 1
    a = function(index_1) - intercept  # x[e1]=1
    index_2[e2] = 1
    b = function(index_2) - intercept  # x[e2]=1
    index_1[e2] = 1
    c = function(index_1) - intercept  # x[e1] and x[e2]=1
    return np.abs(c - (a + b)) < 0.001


# Return True means proper
def check_proper(groups):
    flag = [False] * 1000
    for group in groups:
        for e in group:
            flag[e] = True
    return False not in flag


def groups_feature(groups):
    lens = []
    for g in groups:
        lens.append(len(g))
    print('len of groups: ', len(groups), ' for each group: ', lens)


