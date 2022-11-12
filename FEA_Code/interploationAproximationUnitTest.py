import unittest
import numpy
import mesh
import basis
import interpolationAproximation as inter

class Test_computeSolution( unittest.TestCase ):
    def test_single_linear_element_poly( self ):
        test_solution, node_coords, connect = inter.computeSolution( target_fun = lambda x : x, domain = [-1.0, 1.0 ], num_elems = 1, degree = 1 )
        gold_solution = numpy.array( [ -1.0, 1.0 ] )
        self.assertTrue( numpy.allclose( test_solution, gold_solution ) )

    def test_single_quad_element_poly( self ):
        test_solution, node_coords, connect = inter.computeSolution( target_fun = lambda x : x**2, domain = [-1.0, 1.0 ], num_elems = 1, degree = 2 )
        gold_solution = numpy.array( [ 1.0, 0.0, 1.0 ] )
        self.assertTrue( numpy.allclose( test_solution, gold_solution ) )

    def test_two_linear_element_poly( self ):
        test_solution, node_coords, connect = inter.computeSolution( target_fun = lambda x : x**2, domain = [-1.0, 1.0 ], num_elems = 2, degree = 1 )
        gold_solution = numpy.array( [ 1.0, 0.0, 1.0 ] )
        self.assertTrue( numpy.allclose( test_solution, gold_solution ) )

    def test_four_quad_element_poly( self ):
        test_solution, node_coords, connect = inter.computeSolution( target_fun = lambda x : x**2, domain = [-1.0, 1.0 ], num_elems = 4, degree = 1 )
        gold_solution = numpy.array( [ 1.0, 0.25, 0.0, 0.25, 1.0 ] )
        self.assertTrue( numpy.allclose( test_solution, gold_solution ) )


class Test_evaluateSolutionAt( unittest.TestCase ):
    def test_single_linear_element( self ):
        node_coords, connect = mesh.generateMesh( -1, 1, 1, 1 )
        coeff = numpy.array( [-1.0, 1.0 ] )
        self.assertAlmostEqual( first = inter.evaluateSolutionAt( x = -1.0, coeff = coeff, node_coords = node_coords, connect = connect, eval_basis = basis.evalLagrangeBasis1DanyDomain ), second = -1.0 )
        self.assertAlmostEqual( first = inter.evaluateSolutionAt( x =  0.0, coeff = coeff, node_coords = node_coords, connect = connect, eval_basis = basis.evalLagrangeBasis1DanyDomain  ), second =  0.0 )
        self.assertAlmostEqual( first = inter.evaluateSolutionAt( x = +1.0, coeff = coeff, node_coords = node_coords, connect = connect, eval_basis = basis.evalLagrangeBasis1DanyDomain  ), second = +1.0 )

    def test_two_linear_elements( self ):
        node_coords, connect = mesh.generateMesh( -1, 1, 2, 1 )
        coeff = numpy.array( [ 1.0, 0.0, 1.0 ] )
        self.assertAlmostEqual( first = inter.evaluateSolutionAt( x = -1.0, coeff = coeff, node_coords = node_coords, connect = connect, eval_basis = basis.evalLagrangeBasis1DanyDomain  ), second = +1.0 )
        self.assertAlmostEqual( first = inter.evaluateSolutionAt( x =  0.0, coeff = coeff, node_coords = node_coords, connect = connect, eval_basis = basis.evalLagrangeBasis1DanyDomain  ), second =  0.0 )
        self.assertAlmostEqual( first = inter.evaluateSolutionAt( x = +1.0, coeff = coeff, node_coords = node_coords, connect = connect, eval_basis = basis.evalLagrangeBasis1DanyDomain  ), second = +1.0 )

    def test_single_quadratic_element( self ):
        node_coords, connect = mesh.generateMesh( -1, 1, 1, 2 )
        coeff = numpy.array( [+1.0, 0.0, 1.0 ] )
        self.assertAlmostEqual( first = inter.evaluateSolutionAt( x = -1.0, coeff = coeff, node_coords = node_coords, connect = connect, eval_basis = basis.evalLagrangeBasis1DanyDomain  ), second = +1.0 )
        self.assertAlmostEqual( first = inter.evaluateSolutionAt( x =  0.0, coeff = coeff, node_coords = node_coords, connect = connect, eval_basis = basis.evalLagrangeBasis1DanyDomain  ), second =  0.0 )
        self.assertAlmostEqual( first = inter.evaluateSolutionAt( x = +1.0, coeff = coeff, node_coords = node_coords, connect = connect, eval_basis = basis.evalLagrangeBasis1DanyDomain  ), second = +1.0 )

    def test_two_quadratic_elements( self ):
        node_coords, connect = mesh.generateMesh( -2, 2, 2, 2 )
        coeff = numpy.array( [ 1.0, 0.25, 0.5, 0.25, 1.0 ] )
        self.assertAlmostEqual( first = inter.evaluateSolutionAt( x = -2.0, coeff = coeff, node_coords = node_coords, connect = connect, eval_basis = basis.evalLagrangeBasis1DanyDomain  ), second = +1.00 )
        self.assertAlmostEqual( first = inter.evaluateSolutionAt( x = -1.0, coeff = coeff, node_coords = node_coords, connect = connect, eval_basis = basis.evalLagrangeBasis1DanyDomain  ), second = +0.25 )
        self.assertAlmostEqual( first = inter.evaluateSolutionAt( x =  0.0, coeff = coeff, node_coords = node_coords, connect = connect, eval_basis = basis.evalLagrangeBasis1DanyDomain  ), second = +0.50 )
        self.assertAlmostEqual( first = inter.evaluateSolutionAt( x = +1.0, coeff = coeff, node_coords = node_coords, connect = connect, eval_basis = basis.evalLagrangeBasis1DanyDomain  ), second = +0.25 )
        self.assertAlmostEqual( first = inter.evaluateSolutionAt( x = +2.0, coeff = coeff, node_coords = node_coords, connect = connect, eval_basis = basis.evalLagrangeBasis1DanyDomain  ), second = +1.00 )