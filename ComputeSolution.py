
import unittest
import numpy
import mesh
import basis
import scipy



def getElementIdxContainingPoint(x,node_coords,ien_array):
    num_elems = ien_array.shape[0]
    for elem_idx in range (0, num_elems):
        lower_bound = ien_array[elem_idx,0]
        upper_bound = ien_array[elem_idx, -1]
        node_lower_bound = node_coords[lower_bound]
        node_upper_bound = node_coords[upper_bound]
        if(x >= node_lower_bound and x <= node_upper_bound):
            return elem_idx

def getElementNodes( ien_array, elem_idx ):
    return ien_array[elem_idx]

def getElementDomain( ien_array, node_coords, elem_idx ):
    length = ien_array.shape[1]
    coord1 = node_coords[ien_array[elem_idx,0]]
    coord2 = node_coords[ien_array[elem_idx,length - 1]]
    elem_domain = numpy.array([ coord1 , coord2 ]) 
    return elem_domain

def spatialToParamCoords(x, spatial_domain, param_domain):
    x -= min(spatial_domain)
    jacobian = (param_domain[1]-param_domain[0]) /(spatial_domain[1] - spatial_domain[0])
    x = x*jacobian
    x += min(param_domain)
    return x

def evaluateSolutionAt(x, coeff, node_coords, ien_array, eval_basis):
    param_domain = numpy.array([-1,1])
    if(eval_basis == basis.evalBernsteinBasis1D):
        param_domain = numpy.array([0,1])
        
    elem_idx = getElementIdxContainingPoint(x,node_coords,ien_array)
    elem_nodes = getElementNodes( ien_array, elem_idx )
    elem_domain = getElementDomain( ien_array, node_coords, elem_idx )
    param_coord = spatialToParamCoords(x, elem_domain, param_domain )
    degree = elem_nodes.size -1

    sol_at_point = 0
    for i in range(0,elem_nodes.size):
        curr_node = elem_nodes[i]
        sol_at_point += coeff[curr_node] * eval_basis( param_coord, degree, i )

    return sol_at_point

def computeSolution(target_fun, domain, num_elems, degree):
    node_coords, ien_array = mesh.generateMesh(domain[0], domain[1], num_elems, degree)

    test_solution = target_fun(node_coords)

    return test_solution, node_coords, ien_array

def computeFitError( target_fun, coeff, node_coords, ien_array, eval_basis ):
    num_elems = ien_array.shape[0]
    domain = [ min( node_coords ), max( node_coords ) ]
    abs_err_fun = lambda x : abs( target_fun( x ) - evaluateSolutionAt( x, coeff, node_coords, ien_array, eval_basis ) )
    fit_error, residual = scipy.integrate.quad( abs_err_fun, domain[0], domain[1], epsrel = 1e-12, limit = num_elems * 100 )
    return fit_error, residual


x = 0
spatial_domain = numpy.array([0,1])
param_domain = numpy.array([-1,1])
spatialToParamCoords(x, spatial_domain, param_domain), -1

node_coords, ien_array = mesh.generateMesh( -1, 1, 1, 1 )
coeff = numpy.array( [-1.0, 1.0 ] )
evaluateSolutionAt( x = -1.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D)

class Test_evaluateSolutionAt( unittest.TestCase ):
    def test_spatial_param_coord(self):
        x = 1
        spatial_domain = numpy.array([0,1])
        param_domain = numpy.array([-1,1])
        self.assertAlmostEqual(spatialToParamCoords(x, spatial_domain, param_domain), 1)
        x = 0
        self.assertAlmostEqual(spatialToParamCoords(x, spatial_domain, param_domain), -1)

    def test_get_element_Idx_containing_point(self):
        x = 1
        node_coords, ien_array = mesh.generateMesh( -1, 1, 2, 1 )
        self.assertAlmostEqual(getElementIdxContainingPoint(x,node_coords,ien_array), 1)
        x = -1
        self.assertAlmostEqual(getElementIdxContainingPoint(x,node_coords,ien_array), 0)

    
    def test_single_linear_element( self ):
        node_coords, ien_array = mesh.generateMesh( -1, 1, 1, 1 )
        coeff = numpy.array( [-1.0, 1.0 ] )
        self.assertAlmostEqual( first = evaluateSolutionAt( x = -1.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D ), second = -1.0 )
        self.assertAlmostEqual( first = evaluateSolutionAt( x =  0.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D ), second =  0.0 )
        self.assertAlmostEqual( first = evaluateSolutionAt( x = +1.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D ), second = +1.0 )

    def test_two_linear_elements( self ):
        node_coords, ien_array = mesh.generateMesh( -1, 1, 2, 1 )
        coeff = numpy.array( [ 1.0, 0.0, 1.0 ] )
        self.assertAlmostEqual( first = evaluateSolutionAt( x = -1.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D ), second = +1.0 )
        self.assertAlmostEqual( first = evaluateSolutionAt( x =  0.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D ), second =  0.0 )
        self.assertAlmostEqual( first = evaluateSolutionAt( x = +1.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D ), second = +1.0 )

    def test_single_quadratic_element( self ):
        node_coords, ien_array = mesh.generateMesh( -1, 1, 1, 2 )
        coeff = numpy.array( [+1.0, 0.0, 1.0 ] )
        self.assertAlmostEqual( first = evaluateSolutionAt( x = -1.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D ), second = +1.0 )
        self.assertAlmostEqual( first = evaluateSolutionAt( x =  0.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D ), second =  0.0 )
        self.assertAlmostEqual( first = evaluateSolutionAt( x = +1.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D ), second = +1.0 )

    def test_two_quadratic_elements( self ):
        node_coords, ien_array = mesh.generateMesh( -2, 2, 2, 2 )
        coeff = numpy.array( [ 1.0, 0.25, 0.5, 0.25, 1.0 ] )
        self.assertAlmostEqual( first = evaluateSolutionAt( x = -2.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D ), second = +1.00 )
        self.assertAlmostEqual( first = evaluateSolutionAt( x = -1.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D ), second = +0.25 )
        self.assertAlmostEqual( first = evaluateSolutionAt( x =  0.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D ), second = +0.50 )
        self.assertAlmostEqual( first = evaluateSolutionAt( x = +1.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D ), second = +0.25 )
        self.assertAlmostEqual( first = evaluateSolutionAt( x = +2.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = basis.evalLagrangeBasis1D ), second = +1.00 )

class Test_computeSolution( unittest.TestCase ):
    def test_single_linear_element_poly( self ):
        test_solution, node_coords, ien_array = computeSolution( target_fun = lambda x : x, domain = [-1.0, 1.0 ], num_elems = 1, degree = 1 )
        gold_solution = numpy.array( [ -1.0, 1.0 ] )
        self.assertTrue( numpy.allclose( test_solution, gold_solution ) )

    def test_single_quad_element_poly( self ):
        test_solution, node_coords, ien_array = computeSolution( target_fun = lambda x : x**2, domain = [-1.0, 1.0 ], num_elems = 1, degree = 2 )
        gold_solution = numpy.array( [ 1.0, 0.0, 1.0 ] )
        self.assertTrue( numpy.allclose( test_solution, gold_solution ) )

    def test_two_linear_element_poly( self ):
        test_solution, node_coords, ien_array = computeSolution( target_fun = lambda x : x**2, domain = [-1.0, 1.0 ], num_elems = 2, degree = 1 )
        gold_solution = numpy.array( [ 1.0, 0.0, 1.0 ] )
        self.assertTrue( numpy.allclose( test_solution, gold_solution ) )

    def test_four_quad_element_poly( self ):
        test_solution, node_coords, ien_array = computeSolution( target_fun = lambda x : x**2, domain = [-1.0, 1.0 ], num_elems = 4, degree = 1 )
        gold_solution = numpy.array( [ 1.0, 0.25, 0.0, 0.25, 1.0 ] )
        self.assertTrue( numpy.allclose( test_solution, gold_solution ) )