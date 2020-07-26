import math


def Griewank(x):
    part1 = 0
    part2 = 1
    for i in range(0, len(x)):
        part1 += pow(x[i], 2) / 4000
    for i in range(0, len(x)):
        part2 *= math.cos(x[i] / math.sqrt(i + 1))
    return part1 - part2 + 1


def Schwefel(x):
    part = 418.9829 * len(x)
    for i in range(0, len(x)):
        part -= x[i] * math.sin(math.sqrt(abs(x[i])))
    return part


def Rosenbrock(x):
    part = 0
    for i in range(0, len(x) - 1):
        part += 100 * pow(pow(x[i], 2) - x[i + 1], 2) + pow(x[i] - 1, 2)
    return part


def Rastrigin(x):
    part = 0
    for i in range(0, len(x)):
        part += pow(x[i], 2) - 10 * math.cos(2 * math.pi * x[i]) + 10
    return part


def Ackley(x):
    part1 = 0
    part2 = 0
    for i in range(0, len(x)):
        part1 += pow(x[i], 2)
        part2 += math.cos(2 * math.pi * x[i])
    part1 = -0.2 * math.sqrt(1 / len(x) * part1)
    part2 = 1 / len(x) * part2
    return -20 * math.exp(part1) - math.exp(part2) + 20 + math.e
