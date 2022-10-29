
from cmath import log, sin
from xml.etree.ElementTree import PI
import ComputeSolution
import mesh
import basis
import numpy
import matplotlib.pyplot as plt

target_fun = lambda x : x**3
domain = numpy.array([0.0, 1.0])
num_elems = 2 ** numpy.array( range( 0, 5 ) )
degree = 1
num_nodes = num_elems*(degree+1)
eval_basis = basis.evalLagrangeBasis1D

fit_error=numpy.zeros(num_elems.size)
residual=numpy.zeros(num_elems.size)

for i in range(0,num_elems.size):
    test_solution, node_coords, ien_array = ComputeSolution.computeSolution(target_fun, domain, num_elems[i], degree)
    fit_error[i], residual[i] = ComputeSolution.computeFitError( target_fun, test_solution, node_coords, ien_array, eval_basis )

# target_fun = lambda x : numpy.sin(numpy.pi*x)
# domain = numpy.array([0.0, 1.0])
# num_elems = 2
# degree = 2 ** numpy.array( range( 0, 5 ) )
# num_nodes = num_elems*(degree+1)
# eval_basis = basis.evalLagrangeBasis1D

# fit_error=numpy.zeros(degree.size)
# residual=numpy.zeros(degree.size)

# for i in range(0,degree.size):
#     test_solution, node_coords, ien_array = ComputeSolution.computeSolution(target_fun, domain, num_elems, degree[i])
#     fit_error[i], residual[i] = ComputeSolution.computeFitError( target_fun, test_solution, node_coords, ien_array, eval_basis )

print(num_elems)
print(fit_error)
plt.axes(yscale="log", xscale="log",ylabel = "Error", xlabel = "Number of Elements")
plt.plot(num_elems, fit_error)

plt.show()

plt.axes(yscale="log", xscale="log",ylabel = "Error", xlabel = "Number of Elements")
plt.plot(num_nodes, fit_error)

plt.show()