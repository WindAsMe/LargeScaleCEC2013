import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = [100, 102, 105]

a[:, 1] = 100
print(a)