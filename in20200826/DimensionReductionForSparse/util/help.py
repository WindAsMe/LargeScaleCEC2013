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


def draw_obj(x, y1, y2, y3, name):
    plt.plot(x, y1, label='LASSO Grouping')
    plt.plot(x, y2, label='One Grouping')
    plt.plot(x, y3, label='Random Grouping')
    # plt.plot(x, y4, label='Normal')
    plt.xlabel('Evaluation times')
    plt.ylabel('Fitness')
    plt.legend()
    plt.savefig(
        'D:\CS2019KYUTAI\PythonProject\SparseModeling\in20200826\DimensionReductionForSparse\data\pic\\' + name + '_obj')
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
        'D:\CS2019KYUTAI\PythonProject\SparseModeling\in20200826\DimensionReductionForSparse\data\pic\\' + name + '_var')
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


def write_obj_trace(path, fileName, trace):
    full_path = "D:\CS2019KYUTAI\PythonProject\SparseModeling\in20200826\DimensionReductionForSparse\data\\trace\\obj\\" + path + "\\" + fileName
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


def write_var_trace(path, fileName, trace):
    full_path = "D:\CS2019KYUTAI\PythonProject\SparseModeling\in20200826\DimensionReductionForSparse\data\\trace\\var\\" + path + "\\" + fileName
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
    with open('D:\CS2019KYUTAI\PythonProject\SparseModeling\in20200826\DimensionReductionForSparse\grouping\\' + path, 'w') as file:
        for g in groups:
            file.write(str(g) + ', ')
    file.close()


def F(f=0.5):
    U = random.uniform(0, 1)
    if U < f:
        return abs(random.gauss(0.5, 0.5))
    else:
        return abs(np.random.standard_t(1))


def DE_choice(f=0.5):
    if random.uniform(0, 1) < 0.5:
        return True
    else:
        return False
