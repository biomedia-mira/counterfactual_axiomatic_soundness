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

gamma = Function('gamma', positive=True)(t)
kernel = N / (2 * (N - 1)) * (1 + sqrt(1 - (4 * (N - 1)) / ((N ** 2) * gamma * conjugate(gamma))))
data = Function('data')(x, t)
kde = kernel * gamma * data

primitive = integrate(kde, t)
