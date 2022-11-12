from platform import node
import matplotlib.pyplot as plt
import basis
import mesh
import interpolationAproximation as approxInterp
import basis
import numpy
import matplotlib.pyplot as plt


def plot_fit_error_vs_num_elems():
    target_fun = lambda x: x**3
    domain = [0,1]
    num_elems = 2 ** numpy.array( range( 0, 5 ) )
    degree = 2
    num_nodes = num_elems*(degree+1)
    #num_elems=1
    #degree = 2 ** numpy.array( range( 0, 5 ) )
    eval_basis = basis.evalLagrangeBasis1DanyDomain
    fit_error=numpy.zeros(5)
    residual=numpy.zeros(5)

    #for i in range(0,len(degree)):
    for i in range(0,len(num_elems)):
        solution, node_coords, ien_array = approxInterp.computeSolution( target_fun, domain, num_elems[i], degree)
        fit_error[i], residual[i] = approxInterp.computeFitError( target_fun, solution, node_coords, ien_array, eval_basis )
        
    plt.axes(yscale="log", xscale="log",ylabel = "Error", xlabel = "Number of Elements")
    plt.plot(num_elems, fit_error)
    plt.show()

    plt.axes(yscale="log", xscale="log",ylabel = "Error", xlabel = "Number of Elements")
    plt.plot(num_nodes, fit_error)
    plt.show()

def plot_elem_boundaries(num_elems,node_coords,ien_array):
    for i in range(0, num_elems):
        elem_nodes = ien_array[i]
        elem_domain = [ node_coords[ elem_nodes[0] ], node_coords[ elem_nodes[-1] ] ]
        if i == 0:
            plt.axvline( x = elem_domain[0])
            plt.axvline( x = elem_domain[1] )
        else:
            plt.axvline(x = elem_domain[1])
    
def plot_basis_functions(ien_array,node_coords,coeff,eval_basis):
    num_pts = 100
    num_elems = len( ien_array )
    for e in range( 0, num_elems ):
        elem_nodes = ien_array[e]
        degree = len( elem_nodes ) - 1
        elem_domain = [ node_coords[ elem_nodes[0] ], node_coords[ elem_nodes[-1] ] ]
        x = numpy.linspace( elem_domain[0], elem_domain[1], num_pts )
        y = numpy.zeros( num_pts )
        for n in range( 0, len( elem_nodes ) ):
            curr_node = elem_nodes[n]
            for i in range( 0, num_pts ):
                y[i] = coeff[curr_node] * eval_basis( variate = x[i],degree = degree, basis_idx = n, domain= elem_domain)
            plt.plot(x,y)     

def plotPiecewiseApproximationCoeffs( node_coords, coeff):
    plt.scatter( node_coords, coeff)

def plotFunction( fun, domain):
    x = numpy.linspace( domain[0], domain[1] )          
    plt.plot(x, fun( x ), )

def plotPiecewiseApproximation( ien_array, node_coords, coeff, eval_basis):
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
                y[i] += coeff[curr_node] * eval_basis( variate=x[i],degree = degree, basis_idx = n, domain = elem_domain)           

        plt.plot(x, y)


def plot_basis():
    target_fun = lambda x : numpy.sin(x)
    domain = numpy.array([0.0, numpy.pi*2])
    num_elems = 4
    degree = [1,2,3,4]
    eval_basis = basis.evalBernsteinBasis1DanyDomain
    test_solution, node_coords, ien_array = approxInterp.computeSolution(target_fun, domain, num_elems, degree)
    coeff = numpy.sin( numpy.pi * node_coords )

    plotFunction( target_fun, domain)
    plot_elem_boundaries(num_elems,node_coords,ien_array)
    plot_basis_functions(ien_array,node_coords,coeff,eval_basis)
    plotPiecewiseApproximationCoeffs( node_coords, coeff)
    plotPiecewiseApproximation( ien_array, node_coords, coeff, eval_basis)
    
    plt.show()


plot_basis()
#plot_fit_error_vs_num_elems()