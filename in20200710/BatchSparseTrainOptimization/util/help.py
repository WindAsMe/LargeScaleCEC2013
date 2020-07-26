import random
import copy
import numpy as np
import matplotlib.pyplot as plt


def init_DE_Population(population_size, Dim, scale_range):
    init = []
    for size in range(population_size):
        individual = []
        for D in range(Dim):
            individual.append(random.uniform(scale_range[0], scale_range[1]))
        init.append(individual)
    return np.array(init)


def create_data(scale, Func_Dim, scale_range):
    data = []
    for j in range(scale):
        temp = []
        for i in range(Func_Dim):
            temp.append(random.uniform(scale_range[0], scale_range[1]))
        data.append(temp)
    return np.array(data, dtype='float16')


def create_result(train_data, f, func_num):
    result = []
    for x in train_data:
        result.append(f(x, func_num))
    return np.array(result, dtype='double')


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


def draw_convergence(x, y1, name):
    plt.tight_layout()
    plt.plot(x, y1, label='Grouping')
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.legend()
    plt.savefig('D:\CS2019KYUTAI\PythonProject\SparseModeling\in20200710\BatchSparseTrainOptimization\pic\\' + name + '_convergence')
    plt.show()


def draw_error(y1, name):
    plt.tight_layout()
    x = np.linspace(0, len(y1), len(y1), endpoint=False)
    plt.plot(x, y1, label='coordinate error')
    plt.xlabel('coordinate')
    plt.legend()
    plt.savefig('D:\CS2019KYUTAI\PythonProject\SparseModeling\in20200710\BatchSparseTrainOptimization\pic\\' + name + '_error')
    plt.show()


def write_initial_population(path, name, population):
    f = open('D:\CS2019KYUTAI\PythonProject\SparseModeling\in20200710\BatchSparseTrainOptimization\data\\' + path, 'a')
    f.write(name + '\n')
    f.write(population.__str__() + '\n')
    f.write('\n')
    f.close()


def write_draw(path, name, group_data, normal):
    f = open('D:\CS2019KYUTAI\PythonProject\SparseModeling\in20200710\BatchSparseTrainOptimization\data\\' + path, 'a')
    f.write(name + '\n')
    f.write(group_data.__str__() + '\n')
    f.write(normal.__str__() + '\n')
    f.write('\n')
    f.close()


def write_trace(path, name, round, group_data, normal):
    f = open('D:\CS2019KYUTAI\PythonProject\SparseModeling\in20200710\BatchSparseTrainOptimization\data\\' + path, 'a')
    f.write(name + " " + str(round) + '\n')
    f.write(group_data.__str__() + '\n')
    f.write(normal.__str__() + '\n')
    f.write('\n')
    f.close()


def write_grouping(path, groups):
    with open('D:\CS2019KYUTAI\PythonProject\SparseModeling\in20200710\BatchSparseTrainOptimization\grouping\\' + path, 'w') as file:
        for g in groups:
            file.write(str(g) + ', ')
    file.close()


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
    for i in range(Dim):
        if i not in verify:
            groups_element.append([i])
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



