
import sympy

# Question 33
x,y=sympy.symbols('x y')
func = [11]
for i in range(0,10):
    str_expr = "x**" + str(i)
    expr = sympy.sympify(str_expr)
    func.append(expr)
    print(func[i])

sympy.plot(func[1],func[2],func[3],func[4],func[5],func[6],func[7],func[8],func[9], (x,0,1))

#As the order increases there is smaller and smaller variation which is problematic when dealing with finite precision algebra