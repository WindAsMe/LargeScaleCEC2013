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


def create_data(scale, Func_Dim, Model_Dim, scale_range):
    data = np.zeros((scale, Func_Dim))
    for j in range(0, scale):
        for i in range(0, Model_Dim):
            data[j][i] = random.uniform(scale_range[0], scale_range[1])
    return data


def create_result(train_data, f, func_num):
    result = []
    intercept = f([0] * len(train_data[0]), func_num)
    for x in train_data:
        result.append(f(x, func_num) - intercept)
    return result


# Change the features ['x1', 'x1^2', 'x1 x2^2'] to [[1], [1, 1], [1, 2, 2]]
def feature_names_normalization(feature_names):
    for i in range(len(feature_names)):
        if '^' not in feature_names[i]:
            continue
        feature_split = feature_names[i].split(' ')

        for j in range(len(feature_split)):
            if '^' in feature_split[j]:
                temp = feature_split[j].split('^')
                s = temp[0]
                for time in range(int(temp[1]) - 1):
                    s = s + ' ' + temp[0]
                feature_split[j] = s
        string = feature_split[0]
        for index in range(1, len(feature_split)):
            string = string + ' ' + feature_split[index]
        feature_names[i] = string
    for i in range(len(feature_names)):
        feature_names[i] = feature_names[i].replace('x', '')
    result = []
    for feature in feature_names:
        l = feature.split(' ')
        temp_l = []
        for element in l:
            temp_l.append(int(element))
        result.append(temp_l)
    return result[1:]


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


def group_related_variable(feature_names, max_variable_num):
    temp_feature_names = copy.deepcopy(feature_names)

    index_groups = []
    groups = []
    while temp_feature_names:
        elements = temp_feature_names.pop(0)
        index_group = [feature_names.index(elements)]
        group = [list(set(elements))]
        help_group_related_variable(index_group, group, elements, temp_feature_names, feature_names, len(group[0]), max_variable_num)
        groups.append(group)
        index_groups.append(index_group)
    return groups


def help_group_related_variable(index_group, group, elements, temp_feature_names, feature_names, flag, max_variable_num):
    if flag >= max_variable_num:
        return
    i = -1
    while temp_feature_names:
        i += 1
        if i >= len(temp_feature_names):
            break
        else:
            if have_same_element(elements, temp_feature_names[i]):
                pop = temp_feature_names.pop(i)
                elements = list_combination(elements, pop)
                index_group.append(feature_names.index(pop))
                group.append(pop)
                group = list(set(group))
                i -= 1
                help_group_related_variable(index_group, group, pop, temp_feature_names, feature_names, len(group), max_variable_num)


def group_modified(groups, feature_names):
    new_groups = []
    for group in groups:
        new_group = []
        for element in group:
            new_group += feature_names[element]
        new_groups.append(new_group)
    return new_groups


def extract(groups_modified):
    simple_problems_Data_index = []
    for group in groups_modified:
        simple_problems_Data_index.append(list(set(group)))
    return simple_problems_Data_index


def draw_convergence(x, y1, y2, name):
    plt.tight_layout()
    plt.plot(x, y1, label='Grouping')
    plt.plot(x, y2, label='Normal method')
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.legend()
    plt.savefig('D:\CS2019KYUTAI\PythonProject\SparseModeling\in20200626\DimensionReductionForOriginal\pic\\' + name + '_convergence')
    plt.show()


def draw_error(y1, y2, name):
    plt.tight_layout()
    x = np.linspace(0, len(y1), len(y1), endpoint=False)
    plt.plot(x, y1, label='Already known best coordinate')
    plt.plot(x, y2, label='The best coordinate we calculated')
    plt.xlabel('coordinate')
    plt.legend()
    plt.savefig('D:\CS2019KYUTAI\PythonProject\SparseModeling\in20200626\DimensionReductionForOriginal\pic\\' + name + '_error')
    plt.show()


def write_initial_population(path, name, population):
    f = open('D:\CS2019KYUTAI\PythonProject\SparseModeling\in20200626\DimensionReductionForOriginal\data\\' + path, 'a')
    f.write(name + '\n')
    f.write(population.__str__() + '\n')
    f.write('\n')
    f.close()


def write_draw(path, name, group_data, normal):
    f = open('D:\CS2019KYUTAI\PythonProject\SparseModeling\in20200626\DimensionReductionForOriginal\data\\' + path, 'a')
    f.write(name + '\n')
    f.write(group_data.__str__() + '\n')
    f.write(normal.__str__() + '\n')
    f.write('\n')
    f.close()


def write_trace(path, name, round, group_data, normal):
    f = open('D:\CS2019KYUTAI\PythonProject\SparseModeling\in20200626\DimensionReductionForOriginal\data\\' + path, 'a')
    f.write(name + " " + str(round) + '\n')
    f.write(group_data.__str__() + '\n')
    f.write(normal.__str__() + '\n')
    f.write('\n')
    f.close()



