import numpy
import bext
import basis
import uspline
import quadrature
import scipy
from scipy import integrate
import matplotlib
import matplotlib.pyplot as plt

## MAIN CODE
def computeSolution( problem, uspline_bext ):
    K = assembleStiffnessMatrix(problem, uspline_bext)
    F = assembleForceVector(problem, uspline_bext)
    K, F = applyDisplacement( problem, K, F, uspline_bext )
    #print( "force_vector\n", F )
    d = numpy.matmul(numpy.linalg.inv(K),F)
    d = assembleSolution(d, problem,uspline_bext)
    #print( "coeff\n", d )
    return d

def assembleSolution( coeff, problem, uspline_bext ):
    disp_node_id = bext.getNodeIdNearPoint( uspline_bext, problem[ "displacement" ][ "position" ] )
    coeff = numpy.insert( coeff, disp_node_id, problem[ "displacement" ][ "value" ], axis = 0 )
    return coeff

def applyDisplacement( problem, K, F, uspline_bext ):
    disp_node_id = bext.getNodeIdNearPoint( uspline_bext, problem[ "displacement" ][ "position" ] )
    F -= K[:,disp_node_id] * problem[ "displacement" ][ "value" ]
    K = numpy.delete( numpy.delete( K, disp_node_id, axis = 0 ), disp_node_id, axis = 1 )
    F = numpy.delete( F, disp_node_id, axis = 0 )
    return K, F

def applyTraction( problem, F, uspline_bext ):
    elem_id = bext.getElementIdContainingPoint( uspline_bext, problem[ "traction" ][ "position" ] )
    elem_degree = bext.getElementDegree(uspline_bext,elem_id)
    elem_domain = bext.getElementDomain(uspline_bext,elem_id)
    elem_extract_operator = bext.getElementExtractionOperator(uspline_bext, elem_id)
    global_node_ids = bext.getElementNodeIds(uspline_bext,elem_id)
    for a in range(0,elem_degree + 1):
        A = global_node_ids[a]
        N_a = lambda x: basis.evalSplineBasis1D( elem_extract_operator, a, elem_domain, x)
        F[A] += N_a( problem[ "traction" ][ "position" ] ) * problem[ "traction" ][ "value" ]
    return F

def evaluateConstitutiveModel( problem ):
    return problem[ "elastic_modulus" ] * problem[ "area" ]

def assembleStiffnessMatrix( problem, uspline_bext ):
    basisDeriv = 1
    num_nodes = bext.getNumNodes(uspline_bext)
    num_elems = bext.getNumElems(uspline_bext)
    K = numpy.zeros( shape = ( num_nodes, num_nodes ) )
    for elemIndex in range(0,num_elems):
        elem_id = bext.elemIdFromElemIdx(uspline_bext,elemIndex)
        elem_degree = bext.getElementDegree(uspline_bext,elem_id)
        elem_domain = bext.getElementDomain(uspline_bext,elem_id)
        elem_extract_operator = bext.getElementExtractionOperator(uspline_bext, elem_id)
        num_qp = int( numpy.ceil( ( 2*(elem_degree-basisDeriv) + 1 ) / 2.0 ) )
        global_node_ids = bext.getElementNodeIds(uspline_bext,elem_id)
        for a in range(0,elem_degree+1):      
            A = global_node_ids[a]
            N_a = lambda x: basis.evalSplineBasisDeriv1D( elem_extract_operator, a, basisDeriv, elem_domain, basis.affineMapping1D( [-1, 1], elem_domain, x ))
            for b in range( 0, elem_degree + 1 ):
                B = global_node_ids[b]
                N_b = lambda x: basis.evalSplineBasisDeriv1D( elem_extract_operator, b, basisDeriv, elem_domain, basis.affineMapping1D( [-1, 1], elem_domain, x ))
                integrand = lambda x: N_a(x ) * evaluateConstitutiveModel(problem) * N_b( x )
                K[A, B] += quadrature.quad( integrand, elem_domain, num_qp )
    return K

def assembleForceVector( problem, uspline_bext ):
    num_nodes = bext.getNumNodes(uspline_bext)
    num_elems = bext.getNumElems(uspline_bext)
    F = numpy.zeros(num_nodes)
    for elemIndex in range(0,num_elems):
        elem_id = bext.elemIdFromElemIdx(uspline_bext,elemIndex)
        elem_degree = bext.getElementDegree(uspline_bext,elem_id)
        elem_domain = bext.getElementDomain(uspline_bext,elem_id)
        elem_extract_operator = bext.getElementExtractionOperator(uspline_bext, elem_id)
        num_qp = int( numpy.ceil( ( 2*elem_degree + 1 ) / 2.0 ) )
        global_node_ids = bext.getElementNodeIds(uspline_bext,elem_id)
        for a in range(0,elem_degree+1):
            A = global_node_ids[a]
            N_a = lambda x: basis.evalSplineBasis1D( elem_extract_operator, a, [-1,1], x)
            integrand = lambda x: N_a(x ) * problem[ "body_force" ]
            F[A] += quadrature.quad( integrand, elem_domain, num_qp )
    F = applyTraction( problem, F, uspline_bext )
    return F

## UTILITY CODE
def evaluateSolutionAt( x, coeff, uspline_bext ):
    elem_id = bext.getElementIdContainingPoint( uspline_bext, x )
    elem_nodes = bext.getElementNodeIds( uspline_bext, elem_id )
    elem_domain = bext.getElementDomain( uspline_bext, elem_id )
    elem_degree = bext.getElementDegree( uspline_bext, elem_id )
    elem_extraction_operator = bext.getElementExtractionOperator( uspline_bext, elem_id )
    sol = 0.0
    for n in range( 0, len( elem_nodes ) ):
        curr_node = elem_nodes[n]
        sol += coeff[curr_node] * basis.evalSplineBasis1D( extraction_operator = elem_extraction_operator, basis_idx = n, domain = elem_domain, variate = x )
    return sol

def computeElementFitError( problem, coeff, uspline_bext, elem_id ):
    domain = bext.getDomain( uspline_bext )
    elem_domain = bext.getElementDomain( uspline_bext, elem_id )
    elem_degree = bext.getElementDegree( uspline_bext, elem_id )
    num_qp = int( numpy.ceil( ( 2*(elem_degree - 1) + 1 ) / 2.0 ) + 1 )
    abs_err_fun = lambda x : abs( evaluateExactSolutionAt( problem, basis.affine_mapping_1D( [-1, 1], elem_domain, x ) ) - evaluateSolutionAt( basis.affine_mapping_1D( [-1, 1], elem_domain, x ), coeff, uspline_bext ) )
    abs_error = quadrature.quad( abs_err_fun, elem_domain, num_qp )
    return abs_error

def computeFitError( problem, coeff, uspline_bext ):
    num_elems = bext.getNumElems( uspline_bext )
    abs_error = 0.0
    for elem_idx in range( 0, num_elems ):
        elem_id = bext.elemIdFromElemIdx( uspline_bext, elem_idx )
        abs_error += computeElementFitError( problem, coeff, uspline_bext, elem_id )
    domain = bext.getDomain( uspline_bext )
    target_fun_norm, _ = scipy.integrate.quad( lambda x: abs( evaluateExactSolutionAt( problem, x ) ), domain[0], domain[1], epsrel = 1e-12, limit = num_elems * 100 )
    rel_error = abs_error / target_fun_norm
    return abs_error, rel_error

def plotCompareGoldTestSolution( gold_coeff, test_coeff, uspline_bext ):
    domain = bext.getDomain( uspline_bext )
    x = numpy.linspace( domain[0], domain[1], 1000 )
    yg = numpy.zeros( 1000 )
    yt = numpy.zeros( 1000 )
    for i in range(0, len(x) ):
        yg[i] = evaluateSolutionAt( x[i], test_coeff, uspline_bext )
        yt[i] = evaluateSolutionAt( x[i], gold_coeff, uspline_bext )
    plt.plot( x, yg )
    plt.plot( x, yt )
    plt.show()

def plotCompareFunToExactSolution( problem, test_coeff, uspline_bext ):
    domain = bext.getDomain( uspline_bext )
    x = numpy.linspace( domain[0], domain[1], 1000 )
    ya = numpy.zeros( 1000 )
    ye = numpy.zeros( 1000 )
    for i in range(0, len(x) ):
        ya[i] = evaluateSolutionAt( x[i], test_coeff, uspline_bext )
        ye[i] = evaluateExactSolutionAt( problem, x[i] )
    plt.plot( x, ya )
    plt.plot( x, ye )
    plt.show()

def computeConvergenceRate( num_entities, qoi ):
    def func( x, a, b, c ):
        return a * numpy.power( x, b ) + c
    fit = scipy.optimize.curve_fit(func, num_entities, qoi, method='trf', bounds = ([-numpy.inf, -numpy.inf, -numpy.inf ], [numpy.inf, 0.0, numpy.inf]) )
    a,b,c = fit[0]
    return b

def plotSolution( sol_coeff, uspline_bext ):
    domain = bext.getDomain( uspline_bext )
    x = numpy.linspace( domain[0], domain[1], 1000 )
    y = numpy.zeros( 1000 )
    for i in range(0, len(x) ):
        y[i] = evaluateSolutionAt( x[i], sol_coeff, uspline_bext )
    plt.plot( x, y )
    plt.plot( bext.getSplineNodes( uspline_bext )[:,0], sol_coeff, color = "k", marker = "o", markerfacecolor = "k" )
    plt.show()

def evaluateExactSolutionAt( problem, x ):
    term_1 = problem[ "traction" ][ "value" ] / evaluateConstitutiveModel( problem ) * x
    term_2 = problem[ "displacement" ][ "value" ]
    term_3 =  ( ( problem[ "length" ]**2.0 * problem[ "body_force" ] / 2 ) / evaluateConstitutiveModel( problem ) ) - ( ( ( problem[ "length" ] - x )**2.0 * problem[ "body_force" ] / 2 ) / evaluateConstitutiveModel( problem ) )
    sol = term_1 + term_2 + term_3
    return sol

def plotExactSolution( problem ):
    domain = [0, problem[ "length" ] ]
    x = numpy.linspace( domain[0], domain[1], 1000 )
    y = numpy.zeros( 1000 )
    for i in range(0, len(x) ):
        y[i] = evaluateExactSolutionAt( problem, x[i] )
    plt.plot( x, y )
    plt.show()

# problem = { "elastic_modulus": 100,
#                        "area": 0.01,
#                        "length": 1.0,
#                        "traction": { "value": 1e-3, "position": 1.0 },
#                        "displacement": { "value": 0.0, "position": 0.0 },
#                        "body_force": 1e-3 }
# spline_space = { "domain": [0, problem[ "length" ]], "degree": [ 1, 1, 1 ], "continuity": [ -1, 0, 0, -1 ] }
# uspline.make_uspline_mesh( spline_space, "temp_uspline" )
# uspline_bext = bext.readBEXT( "temp_uspline.json" )
# test_sol_coeff = computeSolution( problem = problem, uspline_bext = uspline_bext )
# plotSolution( test_sol_coeff, uspline_bext )
# plotCompareFunToExactSolution( problem, test_sol_coeff, uspline_bext )

problem = { "elastic_modulus": 200e9,
                       "area": 1.0,
                       "length": 5.0,
                       "traction": { "value": 9810.0, "position": 5.0 },
                       "displacement": { "value": 0.0, "position": 0.0 },
                       "body_force": 784800.0 }
spline_space = { "domain": [0, problem[ "length" ]], "degree": [ 2, 2 ], "continuity": [ -1, 1, -1 ] }
uspline.make_uspline_mesh( spline_space, "temp_uspline" )
uspline_bext = bext.readBEXT( "temp_uspline.json" )
test_sol_coeff = computeSolution( problem = problem, uspline_bext = uspline_bext )
gold_sol_coeff = numpy.array( [0.0, 2.45863125e-05, 4.92339375e-05, 4.92952500e-05] )
plotSolution( test_sol_coeff, uspline_bext )
plotCompareFunToExactSolution( problem, test_sol_coeff, uspline_bext )