import unittest
import uspline
import bext
import splineApproxGalerkin
import numpy
import scipy

target_fun = lambda x: x**0
spline_space = { "domain": [0, 2], "degree": [ 1, 1 ], "continuity": [ -1, 0, -1 ] }
uspline.make_uspline_mesh( spline_space, "temp_uspline" )
uspline_bext = bext.readBEXT( "temp_uspline.json" )
test_gram_matrix = splineApproxGalerkin.assembleGramMatrix( uspline_bext = uspline_bext )
gold_gram_matrix = numpy.array( [ [ 1.0/3.0, 1.0/6.0, 0.0 ],
                                          [ 1.0/6.0, 2.0/3.0, 1.0/6.0 ],
                                          [ 0.0, 1.0/6.0, 1.0/3.0 ] ] )
#self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )

class Test_assembleGramMatrix( unittest.TestCase ):
    def test_two_element_linear_bspline( self ):
        target_fun = lambda x: x**0
        spline_space = { "domain": [0, 2], "degree": [ 1, 1 ], "continuity": [ -1, 0, -1 ] }
        uspline.make_uspline_mesh( spline_space, "temp_uspline" )
        uspline_bext = bext.readBEXT( "temp_uspline.json" )
        test_gram_matrix = splineApproxGalerkin.assembleGramMatrix( uspline_bext = uspline_bext )
        gold_gram_matrix = numpy.array( [ [ 1.0/3.0, 1.0/6.0, 0.0 ],
                                          [ 1.0/6.0, 2.0/3.0, 1.0/6.0 ],
                                          [ 0.0, 1.0/6.0, 1.0/3.0 ] ] )
        self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )

    def test_two_element_quadratic_bspline( self ):
        target_fun = lambda x: x**0
        spline_space = { "domain": [0, 2], "degree": [ 2, 2 ], "continuity": [ -1, 1, -1 ] }
        uspline.make_uspline_mesh( spline_space, "temp_uspline" )
        uspline_bext = bext.readBEXT( "temp_uspline.json" )
        test_gram_matrix = splineApproxGalerkin.assembleGramMatrix( uspline_bext = uspline_bext )
        gold_gram_matrix = numpy.array( [ [ 1.0/5.0, 7.0/60.0, 1.0/60.0, 0.0 ],
                                          [ 7.0/60.0, 1.0/3.0, 1.0/5.0, 1.0/60.0],
                                          [ 1.0/60.0, 1.0/5.0, 1.0/3.0, 7.0/60.0 ],
                                          [ 0.0, 1.0/60.0, 7.0/60.0, 1.0/5.0] ] )
        self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )

    def test_two_element_cubic_bspline( self ):
        spline_space = { "domain": [0, 2], "degree": [ 3, 3 ], "continuity": [ -1, 2, -1 ] }
        uspline.make_uspline_mesh( spline_space, "temp_uspline" )
        uspline_bext = bext.readBEXT( "temp_uspline.json" )
        test_gram_matrix = splineApproxGalerkin.assembleGramMatrix( uspline_bext = uspline_bext )
        gold_gram_matrix = numpy.array( [ [ 1.0/7.0, 7.0/80.0, 1.0/56.0, 1.0/560.0, 0.0 ],
                                          [ 7.0/80.0, 31.0/140.0, 39.0/280.0, 1.0/20.0, 1.0/560.0 ],
                                          [ 1.0/56.0, 39.0/280.0, 13.0/70.0, 39.0/280.0, 1.0/56.0 ],
                                          [ 1.0/560.0, 1.0/20.0, 39.0/280.0, 31.0/140.0, 7.0/80.0 ],
                                          [ 0.0, 1.0/560.0, 1.0/56.0, 7.0/80.0, 1.0/7.0 ] ] )
        self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )


class Test_assembleForceVector( unittest.TestCase ):
    def test_const_force_fun_two_element_linear_bspline( self ):
        target_fun = lambda x: numpy.pi
        spline_space = { "domain": [-1, 1], "degree": [ 1, 1 ], "continuity": [ -1, 0, -1 ] }
        uspline.make_uspline_mesh( spline_space, "temp_uspline" )
        uspline_bext = bext.readBEXT( "temp_uspline.json" )
        test_force_vector = splineApproxGalerkin.assembleForceVector( target_fun = target_fun, uspline_bext = uspline_bext )
        gold_force_vector = numpy.array( [ numpy.pi / 2.0, numpy.pi, numpy.pi / 2.0 ] )
        self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )

    def test_linear_force_fun_two_element_linear_bspline( self ):
        target_fun = lambda x: x
        spline_space = { "domain": [-1, 1], "degree": [ 1, 1 ], "continuity": [ -1, 0, -1 ] }
        uspline.make_uspline_mesh( spline_space, "temp_uspline" )
        uspline_bext = bext.readBEXT( "temp_uspline.json" )
        test_force_vector = splineApproxGalerkin.assembleForceVector( target_fun = target_fun, uspline_bext = uspline_bext )
        gold_force_vector = numpy.array( [ -1.0/3.0, 0.0, 1.0/3.0 ] )
        self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )

    def test_quadratic_force_fun_two_element_linear_bspline( self ):
        target_fun = lambda x: x**2
        spline_space = { "domain": [-1, 1], "degree": [ 1, 1 ], "continuity": [ -1, 0, -1 ] }
        uspline.make_uspline_mesh( spline_space, "temp_uspline" )
        uspline_bext = bext.readBEXT( "temp_uspline.json" )
        test_force_vector = splineApproxGalerkin.assembleForceVector( target_fun = target_fun, uspline_bext = uspline_bext )
        gold_force_vector = numpy.array( [ 1.0/4.0, 1.0/6.0, 1.0/4.0 ] )
        self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )

    def test_const_force_fun_two_element_quadratic_bspline( self ):
        target_fun = lambda x: numpy.pi
        spline_space = { "domain": [-1, 1], "degree": [ 2, 2 ], "continuity": [ -1, 1, -1 ] }
        uspline.make_uspline_mesh( spline_space, "temp_uspline" )
        uspline_bext = bext.readBEXT( "temp_uspline.json" )
        test_force_vector = splineApproxGalerkin.assembleForceVector( target_fun = target_fun, uspline_bext = uspline_bext )
        gold_force_vector = numpy.array( [ numpy.pi/3.0, 2.0*numpy.pi/3.0, 2.0*numpy.pi/3.0, numpy.pi/3.0 ] )
        self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )

    def test_linear_force_fun_two_element_quadratic_bspline( self ):
        target_fun = lambda x: x
        spline_space = { "domain": [-1, 1], "degree": [ 2, 2 ], "continuity": [ -1, 1, -1 ] }
        uspline.make_uspline_mesh( spline_space, "temp_uspline" )
        uspline_bext = bext.readBEXT( "temp_uspline.json" )
        test_force_vector = splineApproxGalerkin.assembleForceVector( target_fun = target_fun, uspline_bext = uspline_bext )
        gold_force_vector = numpy.array( [ -1.0/4.0, -1.0/6.0, 1.0/6.0, 1.0/4.0 ] )
        self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )

    def test_quadratic_force_fun_two_element_quadratic_bspline( self ):
        target_fun = lambda x: x**2
        spline_space = { "domain": [-1, 1], "degree": [ 2, 2 ], "continuity": [ -1, 1, -1 ] }
        uspline.make_uspline_mesh( spline_space, "temp_uspline" )
        uspline_bext = bext.readBEXT( "temp_uspline.json" )
        test_force_vector = splineApproxGalerkin.assembleForceVector( target_fun = target_fun, uspline_bext = uspline_bext )
        gold_force_vector = numpy.array( [ 2.0/10.0, 2.0/15.0, 2.0/15.0, 2.0/10.0 ] )
        self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )

class Test_computeSolution( unittest.TestCase ):
    def test_cubic_polynomial_target_linear_bspline( self ):
        # print( "POLY TEST" )
        target_fun = lambda x: x**3 - (8/5)*x**2 + (3/5)*x
        spline_space = { "domain": [0, 1], "degree": [ 1, 1 ], "continuity": [ -1, 0, -1 ] }
        uspline.make_uspline_mesh( spline_space, "temp_uspline" )
        uspline_bext = bext.readBEXT( "temp_uspline.json" )
        test_sol_coeff = splineApproxGalerkin.computeSolution( target_fun = target_fun, uspline_bext = uspline_bext )
        # plotCompareFunToTestSolution( target_fun, test_sol_coeff, uspline_bext )
        gold_sol_coeff = numpy.array( [ 9.0/160.0, 7.0/240.0, -23.0/480.0 ] )
        abs_err, rel_err = splineApproxGalerkin.computeFitError( target_fun, test_sol_coeff, uspline_bext )
        self.assertTrue( numpy.allclose( gold_sol_coeff, test_sol_coeff ) )
        self.assertAlmostEqual( first = rel_err, second = 0, delta = 1e0 )

    def test_cubic_polynomial_target_quadratic_bspline( self ):
        # print( "POLY TEST" )
        target_fun = lambda x: x**3 - (8/5)*x**2 + (3/5)*x
        spline_space = { "domain": [0, 1], "degree": [ 2, 2 ], "continuity": [ -1, 1, -1 ] }
        uspline.make_uspline_mesh( spline_space, "temp_uspline" )
        uspline_bext = bext.readBEXT( "temp_uspline.json" )
        test_sol_coeff = splineApproxGalerkin.computeSolution( target_fun = target_fun, uspline_bext = uspline_bext )
        # plotCompareFunToTestSolution( target_fun, test_sol_coeff, uspline_bext )
        gold_sol_coeff = numpy.array( [ 1.0/120.0, 9.0/80.0, -1.0/16.0, -1.0/120.0 ] )
        abs_err, rel_err = splineApproxGalerkin.computeFitError( target_fun, test_sol_coeff, uspline_bext )
        self.assertTrue( numpy.allclose( gold_sol_coeff, test_sol_coeff ) )
        self.assertAlmostEqual( first = rel_err, second = 0, delta = 1e-1 )

    def test_cubic_polynomial_target_cubic_bspline( self ):
        # print( "POLY TEST" )
        target_fun = lambda x: x**3 - (8/5)*x**2 + (3/5)*x
        spline_space = { "domain": [0, 1], "degree": [ 3, 3 ], "continuity": [ -1, 2, -1 ] }
        uspline.make_uspline_mesh( spline_space, "temp_uspline" )
        uspline_bext = bext.readBEXT( "temp_uspline.json" )
        test_sol_coeff = splineApproxGalerkin.computeSolution( target_fun = target_fun, uspline_bext = uspline_bext )
        # plotCompareFunToTestSolution( target_fun, test_sol_coeff, uspline_bext )
        gold_sol_coeff = numpy.array( [ 0.0, 1.0/10.0, 1.0/30.0, -1.0/15.0, 0.0 ] )
        abs_err, rel_err = splineApproxGalerkin.computeFitError( target_fun, test_sol_coeff, uspline_bext )
        self.assertTrue( numpy.allclose( gold_sol_coeff, test_sol_coeff ) )
        self.assertAlmostEqual( first = rel_err, second = 0, delta = 1e-12 )

    def test_sin_target( self ):
        # print( "SIN TEST" )
        target_fun = lambda x: numpy.sin( numpy.pi * x )
        spline_space = { "domain": [0, 1], "degree": [ 3, 3 ], "continuity": [ -1, 2, -1 ] }
        uspline.make_uspline_mesh( spline_space, "temp_uspline" )
        uspline_bext = bext.readBEXT( "temp_uspline.json" )
        test_sol_coeff = splineApproxGalerkin.computeSolution( target_fun = target_fun, uspline_bext = uspline_bext )
        abs_err, rel_err = splineApproxGalerkin.computeFitError( target_fun, test_sol_coeff, uspline_bext )
        # plotCompareFunToTestSolution( target_fun, test_sol_coeff, uspline_bext )
        self.assertAlmostEqual( first = rel_err, second = 0, delta = 1e-2 )

    def test_erfc_target( self ):
        # print( "ERFC TEST" )
        target_fun = lambda x: numpy.real( scipy.special.erfc( x ) )
        spline_space = { "domain": [-1, 1], "degree": [ 3, 1, 3 ], "continuity": [ -1, 1, 1, -1 ] }
        uspline.make_uspline_mesh( spline_space, "temp_uspline" )
        uspline_bext = bext.readBEXT( "temp_uspline.json" )
        test_sol_coeff = splineApproxGalerkin.computeSolution( target_fun = target_fun, uspline_bext = uspline_bext )
        abs_err, rel_err = splineApproxGalerkin.computeFitError( target_fun, test_sol_coeff, uspline_bext )
        # plotCompareFunToTestSolution( target_fun, test_sol_coeff, uspline_bext )
        self.assertAlmostEqual( first = rel_err, second = 0, delta = 1e-2 )

    def test_exptx_target( self ):
        # print( "EXPT TEST" )
        target_fun = lambda x: float( numpy.real( float( x )**float( x ) ) )
        spline_space = { "domain": [-1, 1], "degree": [ 5, 5, 5, 5 ], "continuity": [ -1, 4, 0, 4, -1 ] }
        uspline.make_uspline_mesh( spline_space, "temp_uspline" )
        uspline_bext = bext.readBEXT( "temp_uspline.json" )
        test_sol_coeff = splineApproxGalerkin.computeSolution( target_fun = target_fun, uspline_bext = uspline_bext )
        abs_err, rel_err = splineApproxGalerkin.computeFitError( target_fun, test_sol_coeff, uspline_bext )
        # plotCompareFunToTestSolution( target_fun, test_sol_coeff, uspline_bext )
        self.assertAlmostEqual( first = rel_err, second = 0, delta = 1e-2 )