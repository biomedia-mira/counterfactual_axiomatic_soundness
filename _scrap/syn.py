from sympy import IndexedBase, Sum, init_printing, integrate, Function
from sympy.abc import I
from sympy.functions import conjugate, exp, sqrt, sin, cos, exp_polar, polar_lift
import sympy

N = sympy.var('N', real=True, positive=True, integer=True)
x = sympy.var('x', real=True, float=True)
t = sympy.var('t', real=True, float=True)
j = sympy.var('j', real=True, positve=True, integer=True)
X = IndexedBase('X', real=True, float=True)

init_printing()

gamma = (1 / N) * Sum(cos(X[j] * t) + I * sin(X[j] * t), (j, 1, N))
data = cos(x * t) + I * sin(x * t)


gamma = (1 / N) * Sum(exp_polar(X[j] * t), (j, 1, N))
data = cos(x * t) + I * sin(x * t)


kernel = N / (2 * (N - 1)) * (1 + sqrt(1 - (4 * (N - 1)) / ((N ** 2) * gamma * conjugate(gamma))))


f = Function('f')(x, t)

kde = kernel * gamma * data
primitive = integrate(kde, t)
