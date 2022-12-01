import basis
import unittest
import numpy
import math

class Test_evalSplineBasisDeriv1D( unittest.TestCase ):
    def test_C0_linear_0th_deriv_at_nodes( self ):
        C = numpy.eye( 2 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 0, domain = [ 0, 1 ], variate = 0.0 ), second = 1.0 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 0, domain = [ 0, 1 ], variate = 0.0 ), second = 0.0 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 0, domain = [ 0, 1 ], variate = 1.0 ), second = 0.0 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 0, domain = [ 0, 1 ], variate = 1.0 ), second = 1.0 )

    def test_C0_linear_1st_deriv_at_nodes( self ):
        C = numpy.eye( 2 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 1, domain = [ 0, 1 ], variate = 0.0 ), second = -1.0 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 1, domain = [ 0, 1 ], variate = 0.0 ), second = +1.0 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 1, domain = [ 0, 1 ], variate = 1.0 ), second = -1.0 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 1, domain = [ 0, 1 ], variate = 1.0 ), second = +1.0 )

    def test_C1_quadratic_0th_deriv_at_nodes( self ):
        C = numpy.array( [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.5 ], [ 0.0, 0.0, 0.5 ] ] )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 0, domain = [ 0, 1 ], variate = 0.0 ), second = 1.0 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 0, domain = [ 0, 1 ], variate = 0.0 ), second = 0.0 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 0, domain = [ 0, 1 ], variate = 0.0 ), second = 0.0 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 0, domain = [ 0, 1 ], variate = 0.5 ), second = 0.25 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 0, domain = [ 0, 1 ], variate = 0.5 ), second = 0.625 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 0, domain = [ 0, 1 ], variate = 0.5 ), second = 0.125 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 0, domain = [ 0, 1 ], variate = 1.0 ), second = 0.0 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 0, domain = [ 0, 1 ], variate = 1.0 ), second = 0.5 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 0, domain = [ 0, 1 ], variate = 1.0 ), second = 0.5 )

    def test_C1_quadratic_1st_deriv_at_nodes( self ):
        C = numpy.array( [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.5 ], [ 0.0, 0.0, 0.5 ] ] )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 1, domain = [ 0, 1 ], variate = 0.0 ), second = -2.0 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 1, domain = [ 0, 1 ], variate = 0.0 ), second = +2.0 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 1, domain = [ 0, 1 ], variate = 0.0 ), second = +0.0 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 1, domain = [ 0, 1 ], variate = 0.5 ), second = -1.0 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 1, domain = [ 0, 1 ], variate = 0.5 ), second = +0.5 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 1, domain = [ 0, 1 ], variate = 0.5 ), second = +0.5 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 1, domain = [ 0, 1 ], variate = 1.0 ), second = +0.0 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 1, domain = [ 0, 1 ], variate = 1.0 ), second = -1.0 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 1, domain = [ 0, 1 ], variate = 1.0 ), second = +1.0 )

    def test_C1_quadratic_2nd_deriv_at_nodes( self ):
        C = numpy.array( [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.5 ], [ 0.0, 0.0, 0.5 ] ] )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 2, domain = [ 0, 1 ], variate = 0.0 ), second = +2.0 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 2, domain = [ 0, 1 ], variate = 0.0 ), second = -3.0 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 2, domain = [ 0, 1 ], variate = 0.0 ), second = +1.0 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 2, domain = [ 0, 1 ], variate = 0.5 ), second = +2.0 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 2, domain = [ 0, 1 ], variate = 0.5 ), second = -3.0 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 2, domain = [ 0, 1 ], variate = 0.5 ), second = +1.0 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 2, domain = [ 0, 1 ], variate = 1.0 ), second = +2.0 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 2, domain = [ 0, 1 ], variate = 1.0 ), second = -3.0 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 2, domain = [ 0, 1 ], variate = 1.0 ), second = +1.0 )

    def test_biunit_C0_linear_0th_deriv_at_nodes( self ):
        C = numpy.eye( 2 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 0, domain = [ -1, 1 ], variate = -1.0 ), second = 1.0 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 0, domain = [ -1, 1 ], variate = -1.0 ), second = 0.0 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 0, domain = [ -1, 1 ], variate = +1.0 ), second = 0.0 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 0, domain = [ -1, 1 ], variate = +1.0 ), second = 1.0 )

    def test_biunit_C0_linear_1st_deriv_at_nodes( self ):
        C = numpy.eye( 2 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 1, domain = [ -1, 1 ], variate = -1.0 ), second = -0.5 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 1, domain = [ -1, 1 ], variate = -1.0 ), second = +0.5 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 1, domain = [ -1, 1 ], variate = +1.0 ), second = -0.5 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 1, domain = [ -1, 1 ], variate = +1.0 ), second = +0.5 )

    def test_biunit_C1_quadratic_0th_deriv_at_nodes( self ):
        C = numpy.array( [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.5 ], [ 0.0, 0.0, 0.5 ] ] )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 0, domain = [ -1, 1 ], variate = -1.0 ), second = 1.0 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 0, domain = [ -1, 1 ], variate = -1.0 ), second = 0.0 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 0, domain = [ -1, 1 ], variate = -1.0 ), second = 0.0 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 0, domain = [ -1, 1 ], variate = +0.0 ), second = 0.25 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 0, domain = [ -1, 1 ], variate = +0.0 ), second = 0.625 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 0, domain = [ -1, 1 ], variate = +0.0 ), second = 0.125 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 0, domain = [ -1, 1 ], variate = +1.0 ), second = 0.0 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 0, domain = [ -1, 1 ], variate = +1.0 ), second = 0.5 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 0, domain = [ -1, 1 ], variate = +1.0 ), second = 0.5 )

    def test_biunit_C1_quadratic_1st_deriv_at_nodes( self ):
        C = numpy.array( [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.5 ], [ 0.0, 0.0, 0.5 ] ] )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 1, domain = [ -1, 1 ], variate = -1.0 ), second = -1.0 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 1, domain = [ -1, 1 ], variate = -1.0 ), second = +1.0 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 1, domain = [ -1, 1 ], variate = -1.0 ), second = +0.0 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 1, domain = [ -1, 1 ], variate = +0.0 ), second = -0.5 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 1, domain = [ -1, 1 ], variate = +0.0 ), second = +0.25 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 1, domain = [ -1, 1 ], variate = +0.0 ), second = +0.25 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 1, domain = [ -1, 1 ], variate = +1.0 ), second = +0.0 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 1, domain = [ -1, 1 ], variate = +1.0 ), second = -0.5 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 1, domain = [ -1, 1 ], variate = +1.0 ), second = +0.5 )

    def test_biunit_C1_quadratic_2nd_deriv_at_nodes( self ):
        C = numpy.array( [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.5 ], [ 0.0, 0.0, 0.5 ] ] )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 2, domain = [ -1, 1 ], variate = -1.0 ), second = +0.50 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 2, domain = [ -1, 1 ], variate = -1.0 ), second = -0.75 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 2, domain = [ -1, 1 ], variate = -1.0 ), second = +0.25 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 2, domain = [ -1, 1 ], variate = +0.0 ), second = +0.50 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 2, domain = [ -1, 1 ], variate = +0.0 ), second = -0.75 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 2, domain = [ -1, 1 ], variate = +0.0 ), second = +0.25 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 2, domain = [ -1, 1 ], variate = +1.0 ), second = +0.50 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 2, domain = [ -1, 1 ], variate = +1.0 ), second = -0.75 )
        self.assertAlmostEqual( first = basis.evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 2, domain = [ -1, 1 ], variate = +1.0 ), second = +0.25 )

class Test_evalBernsteinBasisDeriv( unittest.TestCase ):
    def test_constant_at_nodes( self ):
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 0, basis_idx = 0, deriv = 0, domain = [0, 1], variate = 0.0 ), second = 1.0, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 0, basis_idx = 0, deriv = 0, domain = [0, 1], variate = 1.0 ), second = 1.0, delta = 1e-12 )

    def test_constant_1st_deriv_at_nodes( self ):
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 0, basis_idx = 0, deriv = 1, domain = [0, 1], variate = 0.0 ), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 0, basis_idx = 0, deriv = 1, domain = [0, 1], variate = 1.0 ), second = 0.0, delta = 1e-12 )

    def test_constant_2nd_deriv_at_nodes( self ):
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 0, basis_idx = 0, deriv = 2, domain = [0, 1], variate = 0.0 ), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 0, basis_idx = 0, deriv = 2, domain = [0, 1], variate = 1.0 ), second = 0.0, delta = 1e-12 )

    def test_linear_at_nodes( self ):
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 1, basis_idx = 0, deriv = 0, domain = [0, 1], variate = 0.0 ), second = 1.0, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 1, basis_idx = 0, deriv = 0, domain = [0, 1], variate = 1.0 ), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 1, basis_idx = 1, deriv = 0, domain = [0, 1], variate = 0.0 ), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 1, basis_idx = 1, deriv = 0, domain = [0, 1], variate = 1.0 ), second = 1.0, delta = 1e-12 )

    def test_linear_at_gauss_pts( self ):
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 1, basis_idx = 0, deriv = 0, domain = [0, 1], variate =  0.5 ), second = 0.5, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 1, basis_idx = 1, deriv = 0, domain = [0, 1], variate =  0.5 ), second = 0.5, delta = 1e-12 )

    def test_quadratic_at_nodes( self ):
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 0, domain = [0, 1], variate = 0.0 ), second = 1.00, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 0, domain = [0, 1], variate = 0.5 ), second = 0.25, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 0, domain = [0, 1], variate = 1.0 ), second = 0.00, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 0, domain = [0, 1], variate = 0.0 ), second = 0.00, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 0, domain = [0, 1], variate = 0.5 ), second = 0.50, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 0, domain = [0, 1], variate = 1.0 ), second = 0.00, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 0, domain = [0, 1], variate = 0.0 ), second = 0.00, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 0, domain = [0, 1], variate = 0.5 ), second = 0.25, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 0, domain = [0, 1], variate = 1.0 ), second = 1.00, delta = 1e-12 )

    def test_quadratic_at_gauss_pts( self ):
        x = [ -1.0 / math.sqrt(3.0) , 1.0 / math.sqrt(3.0) ]
        x = [ basis.affineMapping1D( [-1, 1], [0, 1], xi ) for xi in x ]
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 0, domain = [0, 1], variate = x[0] ), second = 0.62200846792814620, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 0, domain = [0, 1], variate = x[1] ), second = 0.04465819873852045, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 0, domain = [0, 1], variate = x[0] ), second = 0.33333333333333333, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 0, domain = [0, 1], variate = x[1] ), second = 0.33333333333333333, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 0, domain = [0, 1], variate = x[0] ), second = 0.04465819873852045, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 0, domain = [0, 1], variate = x[1] ), second = 0.62200846792814620, delta = 1e-12 )

    def test_linear_1st_deriv_at_nodes( self ):
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 1, basis_idx = 0, deriv = 1, domain = [0, 1], variate = 0.0 ), second = -1.0, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 1, basis_idx = 0, deriv = 1, domain = [0, 1], variate = 1.0 ), second = -1.0, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 1, basis_idx = 1, deriv = 1, domain = [0, 1], variate = 0.0 ), second = +1.0, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 1, basis_idx = 1, deriv = 1, domain = [0, 1], variate = 1.0 ), second = +1.0, delta = 1e-12 )

    def test_linear_1st_deriv_at_gauss_pts( self ):
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 1, basis_idx = 0, deriv = 1, domain = [0, 1], variate = 0.5 ), second = -1.0, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 1, basis_idx = 1, deriv = 1, domain = [0, 1], variate = 0.5 ), second = +1.0, delta = 1e-12 )

    def test_linear_2nd_deriv_at_nodes( self ):
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 1, basis_idx = 0, deriv = 2, domain = [0, 1], variate = 0.0 ), second = 0, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 1, basis_idx = 0, deriv = 2, domain = [0, 1], variate = 1.0 ), second = 0, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 1, basis_idx = 1, deriv = 2, domain = [0, 1], variate = 0.0 ), second = 0, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 1, basis_idx = 1, deriv = 2, domain = [0, 1], variate = 1.0 ), second = 0, delta = 1e-12 )

    def test_linear_2nd_deriv_at_gauss_pts( self ):
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 1, basis_idx = 0, deriv = 2, domain = [0, 1], variate = 0.5 ), second = 0, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 1, basis_idx = 1, deriv = 2, domain = [0, 1], variate = 0.5 ), second = 0, delta = 1e-12 )

    def test_quadratic_1st_deriv_at_nodes( self ):
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 1, domain = [0, 1], variate = 0.0 ), second = -2.0, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 1, domain = [0, 1], variate = 0.5 ), second = -1.0, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 1, domain = [0, 1], variate = 1.0 ), second =  0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 1, domain = [0, 1], variate = 0.0 ), second = +2.0, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 1, domain = [0, 1], variate = 0.5 ), second =  0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 1, domain = [0, 1], variate = 1.0 ), second = -2.0, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 1, domain = [0, 1], variate = 0.0 ), second =  0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 1, domain = [0, 1], variate = 0.5 ), second =  1.0, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 1, domain = [0, 1], variate = 1.0 ), second = +2.0, delta = 1e-12 )

    def test_quadratic_1st_deriv_at_gauss_pts( self ):
        x = [ -1.0 / math.sqrt(3.0) , 1.0 / math.sqrt(3.0) ]
        x = [ basis.affineMapping1D( [-1, 1], [0, 1], xi ) for xi in x ]
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 1, domain = [0, 1], variate = x[0] ), second = -1.0 - 1/( math.sqrt(3) ), delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 1, domain = [0, 1], variate = x[1] ), second = -1.0 + 1/( math.sqrt(3) ), delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 1, domain = [0, 1], variate = x[0] ), second = +2.0 / math.sqrt(3), delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 1, domain = [0, 1], variate = x[1] ), second = -2.0 / math.sqrt(3), delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 1, domain = [0, 1], variate = x[0] ), second = +1.0 - 1/( math.sqrt(3) ), delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 1, domain = [0, 1], variate = x[1] ), second = +1.0 + 1/( math.sqrt(3) ), delta = 1e-12 )

    def test_quadratic_2nd_deriv_at_nodes( self ):
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 1, domain = [0, 1], variate = 0.0 ), second = -2.0, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 1, domain = [0, 1], variate = 0.5 ), second = -1.0, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 1, domain = [0, 1], variate = 1.0 ), second =  0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 1, domain = [0, 1], variate = 0.0 ), second = +2.0, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 1, domain = [0, 1], variate = 0.5 ), second =  0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 1, domain = [0, 1], variate = 1.0 ), second = -2.0, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 1, domain = [0, 1], variate = 0.0 ), second =  0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 1, domain = [0, 1], variate = 0.5 ), second =  1.0, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 1, domain = [0, 1], variate = 1.0 ), second = +2.0, delta = 1e-12 )

    def test_quadratic_2nd_deriv_at_gauss_pts( self ):
        x = [ -1.0 / math.sqrt(3.0) , 1.0 / math.sqrt(3.0) ]
        x = [ basis.affineMapping1D( [-1, 1], [0, 1], xi ) for xi in x ]
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 2, domain = [0, 1], variate = x[0] ), second = +2.0, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 2, domain = [0, 1], variate = x[1] ), second = +2.0, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 2, domain = [0, 1], variate = x[0] ), second = -4.0, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 2, domain = [0, 1], variate = x[1] ), second = -4.0, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 2, domain = [0, 1], variate = x[0] ), second = +2.0, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 2, domain = [0, 1], variate = x[1] ), second = +2.0, delta = 1e-12 )

class Test_evalLegendreBasis1DanyDomain( unittest.TestCase ):
    def test_basisAtBounds( self ):
        for p in range( 0, 2 ):
            if ( p % 2 == 0 ):
                self.assertAlmostEqual( first = basis.evalLegendreBasis1DanyDomain(variate = -1, degree = p, basis_idx = p, domain = [-1, 1]), second = +1.0, delta = 1e-12 )
            else:
                self.assertAlmostEqual( first = basis.evalLegendreBasis1DanyDomain(variate = -1, degree = p, basis_idx = p, domain = [-1, 1]), second = -1.0, delta = 1e-12 )
            self.assertAlmostEqual( first = basis.evalLegendreBasis1DanyDomain(variate = +1, degree = p, basis_idx = p, domain = [-1, 1]), second = 1.0, delta = 1e-12 )
    
    def test_constant( self ):
        for x in numpy.linspace( -1, 1, 100 ):
            self.assertAlmostEqual( first = basis.evalLegendreBasis1DanyDomain(variate = x, degree = 0, basis_idx = 0, domain = [-1, 1]), second = 1.0, delta = 1e-12 )
    
    def test_linear( self ):
        for x in numpy.linspace( -1, 1, 100 ):
            self.assertAlmostEqual( first = basis.evalLegendreBasis1DanyDomain(variate = x, degree = 1, basis_idx = 1, domain = [-1, 1]), second = x, delta = 1e-12 )
    
    def test_quadratic_at_roots( self ):
        self.assertAlmostEqual( first = basis.evalLegendreBasis1DanyDomain(variate = -1.0 / math.sqrt(3.0), degree = 2, basis_idx = 2, domain = [-1, 1]), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalLegendreBasis1DanyDomain(variate = +1.0 / math.sqrt(3.0), degree = 2, basis_idx = 2, domain = [-1, 1]), second = 0.0, delta = 1e-12 )
    
    def test_cubic_at_roots( self ):
        self.assertAlmostEqual( first = basis.evalLegendreBasis1DanyDomain(variate = -math.sqrt( 3 / 5 ), degree = 3, basis_idx = 3, domain = [-1, 1]), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalLegendreBasis1DanyDomain(variate = 0, degree = 3, basis_idx = 3, domain = [-1, 1]), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalLegendreBasis1DanyDomain(variate = +math.sqrt( 3 / 5 ),  degree = 3, basis_idx = 3, domain = [-1, 1]), second = 0.0, delta = 1e-12 )


class Test_evalLegendreBasis1D( unittest.TestCase ):
    def test_basisAtBounds( self ):
        for p in range( 0, 2 ):
            if ( p % 2 == 0 ):
                self.assertAlmostEqual( first = basis.evalLegendreBasis1D( degree = p, variate = -1 ), second = +1.0, delta = 1e-12 )
            else:
                self.assertAlmostEqual( first = basis.evalLegendreBasis1D( degree = p, variate = -1 ), second = -1.0, delta = 1e-12 )
            self.assertAlmostEqual( first = basis.evalLegendreBasis1D( degree = p, variate = +1 ), second = 1.0, delta = 1e-12 )

    def test_constant( self ):
        for x in numpy.linspace( -1, 1, 100 ):
            self.assertAlmostEqual( first = basis.evalLegendreBasis1D( degree = 0, variate = x ), second = 1.0, delta = 1e-12 )

    def test_linear( self ):
        for x in numpy.linspace( -1, 1, 100 ):
            self.assertAlmostEqual( first = basis.evalLegendreBasis1D( degree = 1, variate = x ), second = x, delta = 1e-12 )

    def test_quadratic_at_roots( self ):
        self.assertAlmostEqual( first = basis.evalLegendreBasis1D( degree = 2, variate = -1.0 / math.sqrt(3.0) ), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalLegendreBasis1D( degree = 2, variate = +1.0 / math.sqrt(3.0) ), second = 0.0, delta = 1e-12 )

    def test_cubic_at_roots( self ):
        self.assertAlmostEqual( first = basis.evalLegendreBasis1D( degree = 3, variate = -math.sqrt( 3 / 5 ) ), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalLegendreBasis1D( degree = 3, variate = 0 ), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalLegendreBasis1D( degree = 3, variate = +math.sqrt( 3 / 5 ) ), second = 0.0, delta = 1e-12 )

class Test_evaluateBernsteinBasis1D( unittest.TestCase ):
    def test_linearBernstein( self ):
        self.assertAlmostEqual( first = basis.evalBernsteinBasis1D( variate = -1, degree = 1, basis_idx = 0 ), second = 1.0, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasis1D( variate = -1, degree = 1, basis_idx = 1 ), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasis1D( variate = +1, degree = 1, basis_idx = 0 ), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasis1D( variate = +1, degree = 1, basis_idx = 1 ), second = 1.0, delta = 1e-12 )

    def test_quadraticBernstein( self ):
        self.assertAlmostEqual( first = basis.evalBernsteinBasis1D( variate = -1, degree = 2, basis_idx = 0 ), second = 1.00, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasis1D( variate = -1, degree = 2, basis_idx = 1 ), second = 0.00, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasis1D( variate = -1, degree = 2, basis_idx = 2 ), second = 0.00, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasis1D( variate =  0, degree = 2, basis_idx = 0 ), second = 0.25, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasis1D( variate =  0, degree = 2, basis_idx = 1 ), second = 0.50, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasis1D( variate =  0, degree = 2, basis_idx = 2 ), second = 0.25, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasis1D( variate = +1, degree = 2, basis_idx = 0 ), second = 0.00, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasis1D( variate = +1, degree = 2, basis_idx = 1 ), second = 0.00, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalBernsteinBasis1D( variate = +1, degree = 2, basis_idx = 2 ), second = 1.00, delta = 1e-12 )

class Test_evaluateLagrangeBasis1D( unittest.TestCase ):
    def test_linearLagrange( self ):
        self.assertAlmostEqual( first = basis.evalLagrangeBasis1D( variate = -1, degree = 1, basis_idx = 0 ), second = 1.0, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalLagrangeBasis1D( variate = -1, degree = 1, basis_idx = 1 ), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalLagrangeBasis1D( variate = +1, degree = 1, basis_idx = 0 ), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalLagrangeBasis1D( variate = +1, degree = 1, basis_idx = 1 ), second = 1.0, delta = 1e-12 )

    def test_quadraticLagrange( self ):
        self.assertAlmostEqual( first = basis.evalLagrangeBasis1D( variate = -1, degree = 2, basis_idx = 0 ), second = 1.0, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalLagrangeBasis1D(variate = -1, degree = 2, basis_idx = 1 ), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalLagrangeBasis1D( variate = -1, degree = 2, basis_idx = 2 ), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalLagrangeBasis1D( variate =  0, degree = 2, basis_idx = 0 ), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalLagrangeBasis1D( variate =  0, degree = 2, basis_idx = 1 ), second = 1.0, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalLagrangeBasis1D( variate =  0, degree = 2, basis_idx = 2 ), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalLagrangeBasis1D( variate = +1, degree = 2, basis_idx = 0 ), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalLagrangeBasis1D( variate = +1, degree = 2, basis_idx = 1 ), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = basis.evalLagrangeBasis1D( variate = +1, degree = 2, basis_idx = 2 ), second = 1.0, delta = 1e-12 )