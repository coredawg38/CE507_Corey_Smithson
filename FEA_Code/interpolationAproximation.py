import mesh
import numpy
import basis
import scipy

def computeSolution( target_fun, domain, num_elems, degree):
    if isinstance(degree, int):
        deg = numpy.zeros(num_elems, dtype=int)
        deg.fill(degree)
    else:
        deg =degree
    
    node_coords, ien_array = mesh.generateMeshNonUniformDegree( domain[0], domain[1], deg )
    solution = target_fun( node_coords )

    return solution, node_coords, ien_array

def evaluateSolutionAt( x, coeff, node_coords, connect, eval_basis):
    ien_array = connect
    elem_idx = mesh.getElementIdxContainingPoint(x,node_coords,ien_array)
    elem_nodes = mesh.getElementNodes( ien_array, elem_idx )
    elem_domain = [ node_coords[ elem_nodes[0] ], node_coords[ elem_nodes[-1] ] ]
    degree = len( elem_nodes ) - 1
    sol_at_point = 0.0
    for n in range( 0, len( elem_nodes ) ):
        curr_node = elem_nodes[n]
        sol_at_point += coeff[curr_node] * eval_basis(x, degree, n, elem_domain)
    return sol_at_point

def computeFitError( target_fun, coeff, node_coords, connect, eval_basis ):
    ien_array = connect
    num_elems = len( ien_array )
    domain = [ min( node_coords ), max( node_coords ) ]
    abs_err_fun = lambda x : abs( target_fun( x ) - evaluateSolutionAt( x, coeff, node_coords, connect, eval_basis ) )
    fit_error, residual = scipy.integrate.quad( abs_err_fun, domain[0], domain[1], epsrel = 1e-12, limit = num_elems * 100 )
    return fit_error, residual

test_solution, node_coords, connect = computeSolution( target_fun = lambda x : x, domain = [-1.0, 1.0 ], num_elems = 1, degree = 1 )