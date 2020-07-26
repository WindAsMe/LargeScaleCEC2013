import numpy as np
from sklearn.preprocessing import PolynomialFeatures


def aim_Griewank_original(Phen, reg):
    part1 = 0
    part2 = 1
    for i in range(0, len(Phen[0])):
        part1 += Phen[:, [i]] ** 2 / 4000
        part2 *= np.cos(Phen[:, [i]] / np.sqrt(i + 1))
    return part1 - part2 + 1


def aim_Griewank_one(Phen, reg):
    poly_reg = PolynomialFeatures(degree=1)
    Phen_poly = poly_reg.fit_transform(Phen)
    array = np.array([reg.predict(Phen_poly)]).T
    return array


def aim_Griewank_two(Phen, reg):
    poly_reg = PolynomialFeatures(degree=2)
    Phen_poly = poly_reg.fit_transform(Phen)
    array = np.array([reg.predict(Phen_poly)]).T
    return array


def aim_Schwefel_original(Phen, reg):
    result = 418.9829 * len(Phen[0])
    for i in range(len(Phen[0])):
        result -= Phen[:, [i]] * np.sin(np.sqrt(abs(Phen[:, [i]])))
    return result


def aim_Schwefel_one(Phen, reg):
    poly_reg = PolynomialFeatures(degree=1)
    Phen_poly = poly_reg.fit_transform(Phen)
    array = np.array([reg.predict(Phen_poly)]).T
    return array


def aim_Schwefel_two(Phen, reg):
    poly_reg = PolynomialFeatures(degree=2)
    Phen_poly = poly_reg.fit_transform(Phen)
    array = np.array([reg.predict(Phen_poly)]).T
    return array


def aim_Rosenbrock_original(Phen, reg):
    result = 0
    for i in range(len(Phen[0]) - 1):
        result += (100 * (Phen[:, [i]] ** 2 - Phen[:, [i + 1]]) ** 2 + (Phen[:, [i]] - 1) ** 2)
    return result


def aim_Rosenbrock_one(Phen, reg):
    poly_reg = PolynomialFeatures(degree=1)
    Phen_poly = poly_reg.fit_transform(Phen)
    array = np.array([reg.predict(Phen_poly)]).T
    return array


def aim_Rosenbrock_two(Phen, reg):
    poly_reg = PolynomialFeatures(degree=2)
    Phen_poly = poly_reg.fit_transform(Phen)
    array = np.array([reg.predict(Phen_poly)]).T
    return array


def aim_Rastrigin_original(Phen, reg):
    result = 0
    for i in range(len(Phen[0])):
        result += (Phen[:, [i]] ** 2 - 10 * np.cos(2 * np.pi * Phen[:, [i]]) + 10)
    return result


def aim_Rastrigin_one(Phen, reg):
    poly_reg = PolynomialFeatures(degree=1)
    Phen_poly = poly_reg.fit_transform(Phen)
    array = np.array([reg.predict(Phen_poly)]).T
    return array


def aim_Rastrigin_two(Phen, reg):
    poly_reg = PolynomialFeatures(degree=2)
    Phen_poly = poly_reg.fit_transform(Phen)
    array = np.array([reg.predict(Phen_poly)]).T
    return array


def aim_Ackley_original(Phen, reg):
    part1 = 0
    part2 = 0
    for i in range(len(Phen[0])):
        part1 += Phen[:, [i]] ** 2
        part2 += np.cos(2 * np.pi * Phen[:, [i]])
    return -20 * np.exp(-0.2 * np.sqrt(part1 / len(Phen[0]))) - np.exp(part2 / len(Phen[0])) + 20 + np.e


def aim_Ackley_one(Phen, reg):
    poly_reg = PolynomialFeatures(degree=1)
    Phen_poly = poly_reg.fit_transform(Phen)
    array = np.array([reg.predict(Phen_poly)]).T
    return array


def aim_Ackley_two(Phen, reg):
    poly_reg = PolynomialFeatures(degree=2)
    Phen_poly = poly_reg.fit_transform(Phen)
    array = np.array([reg.predict(Phen_poly)]).T
    return array


