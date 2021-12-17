import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.optimize as opt


def y1(_x):
    return (np.arctan(_x ** 2) - 0.3) / _x


def y2(_x):
    return np.sqrt((1 - 0.9 * _x ** 2) / 2)


def fu(p):
    _x, _y = p
    return np.tan(_x * _y + 0.3) - _x ** 2, 0.9 * _x ** 2 + 2 * _y ** 2 - 1


def func1(_x, _y):
    return np.tan(_x * _y + 0.3) - _x ** 2


def func2(_x, _y):
    return 0.9 * _x ** 2 + 2 * _y ** 2 - 1


def func1_x(_x, _y):
    return -2 * _x + _y * (np.tan(_x * _y + 0.3) ** 2 + 1)


def func1_y(_x, _y):
    return _x * (np.tan(_x * _y + 0.3) ** 2 + 1)


def func2_x(_x, _y):
    return 1.8 * _x


def func2_y(_x, _y):
    return 4 * _y


def alpha(_x, _y):
    return 1 / (func1_y(_x, _y) * func2_x(_x, _y) / func2_y(_x, _y) - func1_x(_x, _y))


def beta(_x, _y):
    return func1_y(_x, _y) / (func1_x(_x, _y) * func2_y(_x, _y) - func1_y(_x, _y) * func2_x(_x, _y))


def gamma(_x, _y):
    return 1 / (func1_x(_x, _y) * func2_y(_x, _y) / func2_x(_x, _y) - func1_y(_x, _y))


def delta(_x, _y):
    return func1_x(_x, _y) / (func2_x(_x, _y) * func1_y(_x, _y) - func1_x(_x, _y) * func2_y(_x, _y))


def iterF1(_x, _y):
    return _x + alpha(_x, _y) * func1(_x, _y) + beta(_x, _y) * func2(_x, _y)


def iterF2(_x, _y):
    return _y + gamma(_x, _y) * func1(_x, _y) + delta(_x, _y) * func2(_x, _y)


def simple_iteration_method(_x, _y, _eps, f1, f2):
    i = 0
    while math.sqrt((f1(_x, _y) - _x) ** 2 + (f2(_x, _y) - _y) ** 2) >= _eps:
        i += 1
        _x = f1(_x, _y)
        _y = f2(_x, _y)
    return _x, _y, i


def seidel_method(_x, y, _eps, f1, f2):
    i = 0
    while math.sqrt((f1(_x, y) - _x) ** 2 + (f2(f1(_x, y), y) - y) ** 2) >= _eps:
        i += 1
        _x = f1(_x, y)
        y = f2(f1(_x, y), y)
    return _x, y, i


def W(_x, _y):
    M = [[0] * 2 for _ in range(2)]
    M[0][0] = func1_x(_x, _y)
    M[1][0] = func2_x(_x, _y)
    M[0][1] = func1_y(_x, _y)
    M[1][1] = func2_y(_x, _y)
    return M


def A1(_x, _y):
    M = [[0] * 2 for _ in range(2)]

    M[0][0] = func1(_x, _y)
    M[1][0] = func2(_x, _y)
    M[0][1] = func1_y(_x, _y)
    M[1][1] = func2_y(_x, _y)
    return M


def A2(_x, _y):
    M = [[0] * 2 for _ in range(2)]
    M[0][0] = func1_x(_x, _y)
    M[1][0] = func2_x(_x, _y)
    M[0][1] = func1(_x, _y)
    M[1][1] = func2(_x, _y)
    return M


def func1_newton(_x, _y):
    return _x - np.linalg.det(A1(_x, _y)) / np.linalg.det(W(_x, _y))


def func2_newton(_x, _y):
    return _y - np.linalg.det(A2(_x, _y)) / np.linalg.det(W(_x, _y))


def print_result(sol):
    print("%0.5f %0.5f" % (sol[0], sol[1]))
    print("Amount of iteration = " + str(sol[2]))
    print("")


eps = 10e-15
x0 = 0.8
y0 = 0.4

sol1 = simple_iteration_method(x0, y0, eps, iterF1, iterF2)
sol2 = seidel_method(x0, y0, eps, iterF1, iterF2)
sol3 = simple_iteration_method(x0, y0, eps, func1_newton, func2_newton)

print("Simple Iteration solution ")
print_result(sol1)

print("Seidel solution ")
print_result(sol2)

print("Newton solution ")
print_result(sol3)

x_sci = opt.fsolve(fu, (1, 1))[0]
print("SciPy solution")
print(x_sci, y1(x_sci))

x = np.arange(0.1, 1, 0.01)
plt.plot(x, y1(x), x, y2(x), "-")
plt.vlines(x_sci, -3, 1, colors="r")
plt.scatter(x_sci, y1(x_sci))
plt.grid()
plt.show()
print(func1_x(x_sci, y1(x_sci)))
