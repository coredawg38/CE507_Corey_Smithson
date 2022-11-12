import numpy
import mesh
import quadrature
import basis
import scipy



def assembleGramMatrix( node_coords, ien_array, solution_basis):
    num_nodes = len( node_coords )
    num_elems = len( ien_array )
    M = numpy.zeros( shape = ( num_nodes, num_nodes ) )
    for elemIndex in range(0,num_elems):
        elem_degree = len( ien_array[elemIndex]) -1
        elem_domain = mesh.getElemDomain(node_coords, ien_array, elemIndex)
        num_qp = int( numpy.ceil( ( 2*elem_degree + 1 ) / 2.0 ) )
        for a in range(0,elem_degree+1):
            A = mesh.getGlobalNodeID(ien_array,elemIndex,a)
            N_a= lambda x:  solution_basis(x,elem_degree,a,[-1,1])
            for b in range( 0, elem_degree + 1 ):
                B = mesh.getGlobalNodeID(ien_array,elemIndex,b)
                N_b = lambda x: solution_basis(x, elem_degree, b, [-1, 1])
                integrand = lambda x: N_a(x ) * N_b( x )
                M[A, B] += quadrature.quad( integrand, elem_domain, num_qp )
    return M


def assembleForceVector( target_fun, node_coords, ien_array, solution_basis):
    num_nodes = len( node_coords )
    num_elems = len( ien_array )
    F = numpy.zeros(num_nodes)
    for elemIndex in range(0,num_elems):
        elem_degree = len( ien_array[elemIndex]) -1
        elem_domain = mesh.getElemDomain(node_coords, ien_array, elemIndex)
        num_qp = int( numpy.ceil( ( 2*elem_degree + 1 ) / 2.0 ) )
        for a in range(0,elem_degree+1):
            A = mesh.getGlobalNodeID(ien_array,elemIndex,a)
            N_a= lambda x:  solution_basis(x,elem_degree,a,[-1,1])
            integrand = lambda x: N_a(x ) * target_fun( basis.affineMapping1D( [-1, 1], elem_domain, x) )
            F[A] += quadrature.quad( integrand, elem_domain, num_qp )
    return F

def computeSolution(target_fun, domain, degree, solution_basis):
    node_coords, ien_array = mesh.generateMeshNonUniformDegree( domain[0], domain[1], degree )
    M = assembleGramMatrix( node_coords, ien_array, solution_basis)
    F = assembleForceVector( target_fun, node_coords, ien_array, solution_basis)
    d = numpy.matmul(numpy.linalg.inv(M),F)
    return d, node_coords, ien_array

def computeFitError( target_fun, test_sol_coeff, node_coords, ien_array, solution_basis):
    
    num_elems = len( ien_array )
    domain = [ min( node_coords ), max( node_coords ) ]
    abs_err_fun = lambda x : abs( target_fun( x ) - evaluateSolutionAt( x, test_sol_coeff, node_coords, ien_array, solution_basis ) )
    fit_error, residual = scipy.integrate.quad( abs_err_fun, domain[0], domain[1], epsrel = 1e-12, limit = num_elems * 100 )
    return fit_error, residual

def evaluateSolutionAt( x, coeff, node_coords, ien_array, eval_basis):
    elem_idx = mesh.getElementIdxContainingPoint(x,node_coords,ien_array)
    elem_nodes = mesh.getElementNodes( ien_array, elem_idx )
    elem_domain = [ node_coords[ elem_nodes[0] ], node_coords[ elem_nodes[-1] ] ]
    degree = len( elem_nodes ) - 1
    sol_at_point = 0.0
    for n in range( 0, len( elem_nodes ) ):
        curr_node = elem_nodes[n]
        sol_at_point += coeff[curr_node] * eval_basis(x, degree, n, elem_domain)
    return sol_at_point


