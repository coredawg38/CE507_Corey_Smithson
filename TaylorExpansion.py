
from cmath import log
import sympy
import scipy
import matplotlib.pyplot as plt

#Question 29
def taylorExpansion( fun, a, order ):
    x = list( fun.atoms( sympy.Symbol ) )[0]
    t = 0
    for i in range( 0, order + 1 ):
        df = sympy.diff( fun, x, i )
        term = ( df.subs( x, a ) / sympy.factorial( i ) ) * ( x - a )**i
        t += term
    return t

x,y=sympy.symbols('x y')

#Question 30

#sin(pi*x) comparison
str_expr = "sin(pi*x)"
expr = sympy.sympify(str_expr)

order0 = taylorExpansion(expr,0,0)
order1 = taylorExpansion(expr,0,1)
order3 = taylorExpansion(expr,0,3)
order5 = taylorExpansion(expr,0,5)
order7 = taylorExpansion(expr,0,7)


sympy.plot(expr,order0,order1,order3,order5,order7, (x,-1,1))

#exp(x) comparison
str_expr = "exp(x)"
expr = sympy.sympify(str_expr)

order0 = taylorExpansion(expr,0,0)
order1 = taylorExpansion(expr,0,1)
order2 = taylorExpansion(expr,0,2)
order3 = taylorExpansion(expr,0,3)
order4 = taylorExpansion(expr,0,4)


sympy.plot(expr,order0,order1,order2,order3,order4, (x,-1,1))

#erfc(x) Comparison
str_expr = "erfc(x)"
expr = sympy.sympify(str_expr)

order0 = taylorExpansion(expr,0,0)
order1 = taylorExpansion(expr,0,1)
order3 = taylorExpansion(expr,0,3)
order5 = taylorExpansion(expr,0,5)
order7 = taylorExpansion(expr,0,7)
order9 = taylorExpansion(expr,0,9)
order11 = taylorExpansion(expr,0,11)

sympy.plot(expr,order0,order1,order3,order5,order7,order9, order11, (x,-2,2))

#Questions 31-32

#10 term taylor series expansion of sin(pi*x)
str_expr = "sin(pi*x)"
expr = sympy.sympify(str_expr)
intr = sympy.integrate(expr)
taylor_expansion = taylorExpansion(expr,0,10)
taylor_expansion_integral = sympy.integrate(taylor_expansion,x)
print(intr)
print(taylor_expansion_integral)

error = []
degree = list( range( 0, 11 ) )
for p in degree:
    t = taylorExpansion( expr, 0, p )
    error.append( scipy.integrate.quad( sympy.lambdify( x, abs( t - expr ) ), -1, 1, limit = 1000 )[0] )
print(error)
plt.plot(degree,error)
plt.yscale("log")
plt.show()

#10 term taylor series expansion of exp(x)
str_expr = "exp(x)"
expr = sympy.sympify(str_expr)
intr = sympy.integrate(expr)
order10 = taylorExpansion(expr,0,10)
order10 = sympy.integrate(order10,x)
print(intr)
print(order10)

error = []
degree = list( range( 0, 11 ) )
for p in degree:
    t = taylorExpansion( expr, 0, p )
    error.append( scipy.integrate.quad( sympy.lambdify( x, abs( t - expr ) ), -1, 1, limit = 1000 )[0] )
print(error)
plt.plot(degree,error)
plt.yscale("log")
plt.show()


#10 term taylor series expansion of erfc(x)
str_expr = "erfc(x)"
expr = sympy.sympify(str_expr)
intr = sympy.integrate(expr)
order10 = taylorExpansion(expr,0,10)
order10 = sympy.integrate(order10,x)
print(intr)
print(order10)

error = []
degree = list( range( 0, 11 ) )
for p in degree:
    t = taylorExpansion( expr, 0, p )
    error.append( scipy.integrate.quad( sympy.lambdify( x, abs( t - expr ) ), -2, 2, limit = 1000 )[0] )
print(error)
plt.plot(degree,error)
plt.yscale("log")
plt.show()