from platform import node
import matplotlib.pyplot as plt
import ComputeSolution
import mesh
import basis
import numpy
import matplotlib.pyplot as plt


def plot_node_boundaries(num_elems,node_coords,ien_array):
    for i in range(1, num_elems):
        x_val = node_coords[ien_array[i,0]]
        plt.axvline(x_val)
    
def plot_basis_functions(node_coords,ien_array,coeff,eval_basis):
    num_pts = 100
    num_elems = len( ien_array )
    for e in range( 0, num_elems ):
        elem_nodes = ien_array[e]
        degree = len( elem_nodes ) - 1
        elem_domain = [ node_coords[ elem_nodes[0] ], node_coords[ elem_nodes[-1] ] ]
        param_domain = [-1,1]
        x = numpy.linspace( elem_domain[0], elem_domain[1], num_pts )
        y = numpy.zeros( num_pts )
        for n in range( 0, len( elem_nodes ) ):
            curr_node = elem_nodes[n]
            for i in range( 0, num_pts ):
                param_coord = ComputeSolution.spatialToParamCoords(x[i], elem_domain, param_domain )
                y[i] = coeff[curr_node] * eval_basis( variate=param_coord,degree = degree, basis_idx = n)
            plt.plot(x,y)     

def plotPiecewiseApproximationCoeffs( node_coords, coeff):
    plt.scatter( node_coords, coeff)

def plotFunction( fun, domain):
    x = numpy.linspace( domain[0], domain[1] )          
    plt.plot(x, fun( x ), )

def plotPiecewiseApproximation( ien_array, node_coords, coeff, eval_basis):
    param_domain = [-1,1]
    num_pts = 100
    num_elems = len( ien_array )
    for e in range( 0, num_elems ):
        elem_nodes = ien_array[e]
        degree = len( elem_nodes ) - 1
        elem_domain = [ node_coords[ elem_nodes[0] ], node_coords[ elem_nodes[-1] ] ]
        x = numpy.linspace( elem_domain[0], elem_domain[1], num_pts )
        y = numpy.zeros( num_pts )
        for i in range( 0, num_pts ):
            for n in range( 0, len( elem_nodes ) ):
                curr_node = elem_nodes[n]
                param_coord = ComputeSolution.spatialToParamCoords(x[i], elem_domain, param_domain )
                y[i] += coeff[curr_node] * eval_basis( variate=param_coord,degree = degree, basis_idx = n)           

        plt.plot(x, y)


def plot_basis():
    target_fun = lambda x : numpy.sin(numpy.pi*x)
    domain = numpy.array([0.0, 4.0])
    num_elems = 3
    degree = 2
    num_nodes = num_elems*(degree+1)
    eval_basis = basis.evalLagrangeBasis1D
    test_solution, node_coords, ien_array = ComputeSolution.computeSolution(target_fun, domain, num_elems, degree)
    coeff = numpy.sin( numpy.pi * node_coords )

    #plotFunction( target_fun, domain)
    plot_node_boundaries(num_elems, node_coords, ien_array)
    plot_basis_functions(node_coords,ien_array,coeff,eval_basis)
    plotPiecewiseApproximationCoeffs( node_coords, coeff)
    plotPiecewiseApproximation( ien_array, node_coords, coeff, eval_basis)
    
    plt.show()

plot_basis()