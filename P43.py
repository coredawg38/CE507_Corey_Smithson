import sympy
import LagrangeBasis
import BernsteinBasis
import LegendreBasis
import numpy


x,y=sympy.symbols('x y')
expr = x**1 + x**0 + x**2 + x**3

sympy.plot(sympy.sympify(x**0),sympy.sympify(x**1),sympy.sympify(x**2),sympy.sympify(x**3),sympy.sympify(expr), (x,-1,1))

bern1 = [200]
bern2 = [200]
x=-1
for i in range(0,200):
    x += .01
    bern1[i] = 0*LagrangeBasis.evaluateLagrangeBasis1D(x,0,3)
    bern2[i] = 4*LagrangeBasis.evaluateLagrangeBasis1D(x,1,3)/3

sympy.plot(bern1, bern2, (x,-1,1))