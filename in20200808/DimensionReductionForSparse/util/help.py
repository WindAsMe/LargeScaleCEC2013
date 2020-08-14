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


def create_data(scale, dim, scale_range):
    data = []
    for j in range(0, scale):
        temp = []
        for i in range(0, dim):
            temp.append(random.uniform(scale_range[0], scale_range[1]))
        data.append(temp)

    return np.array(data, dtype='float16')


def create_result(train_data, f):
    result = []
    for x in train_data:
        result.append(f(x))
    return result


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


def group_related_variable(feature_names):
    temp_feature_names = copy.deepcopy(feature_names)

    groups = []
    while temp_feature_names:
        elements = temp_feature_names.pop(0)
        group = [feature_names.index(elements)]
        help_group_related_variable(group, elements, temp_feature_names, feature_names)
        groups.append(group)
    return groups


def help_group_related_variable(group, elements, temp_feature_names, feature_names):
    i = -1
    while temp_feature_names:
        i += 1
        if i >= len(temp_feature_names):
            break
        else:
            if have_same_element(elements, temp_feature_names[i]):
                pop = temp_feature_names.pop(i)
                elements = list_combination(elements, pop)
                group.append(feature_names.index(pop))
                i -= 1
                help_group_related_variable(group, pop, temp_feature_names, feature_names)


def group_modified(groups, feature_names):
    new_groups = []
    for group in groups:
        new_group = []
        for element in group:
            new_group += feature_names[element]
        new_groups.append(new_group)
    return new_groups


def extract(groups_modified):
    simple_problems_Dim = []
    simple_problems_Data_index = []
    for group in groups_modified:
        simple_problems_Dim.append(len(set(group)))
        simple_problems_Data_index.append(list(set(group)))
    return simple_problems_Dim, simple_problems_Data_index


def draw_obj(x1, x2, y1, y4, name):
    plt.plot(x1, y1, label='LASSO Grouping')
    # plt.plot(x1, y3, label='One Grouping')
    plt.plot(x2, y4, label='Normal')
    plt.xlabel('Evaluation times')
    plt.ylabel('Fitness')
    plt.legend()
    plt.savefig(
        'D:\CS2019KYUTAI\PythonProject\SparseModeling\in20200808\DimensionReductionForSparse\data\pic\\' + name + '_obj')
    plt.show()


def draw_var(x, y1, y4, name):
    plt.tight_layout()
    plt.scatter(x, y1, label='LASSO Grouping', marker='.', c='c')
    # plt.scatter(x, y2, label='Random Grouping', marker=',', c='g')
    # plt.scatter(x, y3, label='One Grouping', marker='o', c='k')
    plt.plot(x, y4, label='Known')
    plt.xlabel('coordinate')
    plt.ylabel('value')
    plt.legend()
    plt.savefig(
        'D:\CS2019KYUTAI\PythonProject\SparseModeling\in20200808\DimensionReductionForSparse\data\pic\\' + name + '_var')
    plt.show()


def groups_one_create(Dim):
    groups = []
    for i in range(Dim):
        groups.append([i])
    return groups


def groups_random_create(Dim, groups_num=25, max_number=10):
    groups = []
    d = {}
    for i in range(Dim):
        d[i] = []
    for i in range(Dim):
        r = random.randint(0, groups_num-1)
        while len(d[r]) >= max_number:
            r = random.randint(0, groups_num-1)
        d[r].append(i)
    for i in d:
        if d[i]:
            groups.append(d[i])
    return groups


def write_trace(fileName, trace):
    full_path = "D:\CS2019KYUTAI\PythonProject\SparseModeling\in20200808\DimensionReductionForSparse\data\\trace\\" + fileName
    with open(full_path, 'a') as f:
        f.write('[')
        for i in range(len(trace)):
            if i == len(trace) - 1:
                f.write(str(trace[i]))
            else:
                f.write(str(trace[i]) + ', ')
        f.write(']')
        f.write('\n')
        f.close()
