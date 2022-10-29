from platform import node
import matplotlib.pyplot as plt
import ComputeSolution
import mesh
import basis
import numpy
import matplotlib.pyplot as plt

target_fun = lambda x : x**3
domain = numpy.array([0.0, 1.0])
num_elems = 3
degree = 1
num_nodes = num_elems*(degree+1)
eval_basis = basis.evalLagrangeBasis1D
test_solution, node_coords, ien_array = ComputeSolution.computeSolution(target_fun, domain, num_elems, degree)

for i in range(1, num_elems):
    x_val = node_coords[ien_array[i,0]]
    plt.axvline(x_val)



coeff = numpy.array( [-1.0, 1.0 ] )
sol_at_point = ComputeSolution.evaluateSolutionAt( x = -1.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D)
plt.show()


# The (scaled) basis functions of the piecewise approximation

# The coefficients of the piecewise approximation

# The piecewise approximation