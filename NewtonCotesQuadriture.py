import unittest
import math
import numpy

def computeNewtonCotesQuadrature( fun, num_points):
    pointVec, weightVec = getNewtonCotesQuadrature(num_points=num_points)

    value = 0
    for i in range (0,num_points):
        value += fun(pointVec[i])*weightVec[i]

    return value

def getNewtonCotesQuadrature( num_points):
    if num_points == 1:
        pointVec = numpy.array([0])
        weightVec = numpy.array([2])
    elif num_points ==2:
        pointVec = numpy.array([-1,1])
        weightVec = numpy.array([1,1])
    elif num_points ==3:
        pointVec = numpy.array([-1,0,1])
        weightVec = numpy.array([1/3,4/3,1/3])
    elif num_points ==4:
        pointVec = numpy.array([-1,-1/3,1/3,1])
        weightVec = numpy.array([1/4,3/4,3/4,1/4])
    elif num_points ==5:
        pointVec = numpy.array([-1,-1/2,0,1/2,1])
        weightVec = numpy.array([7/45,32/45,4/15,32/45,7/45])
    elif num_points ==6:
        pointVec = numpy.array([-1,-3/5,-1/5,1/5,3/5,1])
        weightVec = numpy.array([19/144,25/48,25/72,25/72,25/48,19/144])
    else:
        raise Exception("num_points_MUST_BE_INTEGER_IN_[1,6]")
    return pointVec, weightVec

class Test_computeNewtonCotesQuadrature( unittest.TestCase ):
    def test_integrate_constant_one( self ):
        constant_one = lambda x : 1 * x**0
        for degree in range( 1, 6 ):
            num_points = degree + 1
            self.assertAlmostEqual( first = computeNewtonCotesQuadrature( fun = constant_one, num_points = num_points ), second = 2.0, delta = 1e-12 )

    def test_exact_poly_int( self ):
        for degree in range( 1, 6 ):
            num_points = degree + 1
            poly_fun = lambda x : ( x + 1.0 ) ** degree
            indef_int = lambda x : ( ( x + 1 ) ** ( degree + 1) ) / ( degree + 1 )
            def_int = indef_int(1.0) - indef_int(-1.0)
            self.assertAlmostEqual( first = computeNewtonCotesQuadrature( fun = poly_fun, num_points = num_points ), second = def_int, delta = 1e-12 )

    def test_integrate_sin( self ):
        sin = lambda x : math.sin(x)
        for num_points in range( 1, 7 ):
            self.assertAlmostEqual( first = computeNewtonCotesQuadrature( fun = sin, num_points = num_points ), second = 0.0, delta = 1e-12 )

    def test_integrate_cos( self ):
        cos = lambda x : math.cos(x)
        self.assertAlmostEqual( first = computeNewtonCotesQuadrature( fun = cos, num_points = 6 ), second = 2*math.sin(1), delta = 1e-4 )


class Test_getNewtonCotesQuadrature( unittest.TestCase ):
    def test_incorrect_num_points( self ):
        with self.assertRaises( Exception ) as context:
            getNewtonCotesQuadrature( num_points = 0 )
        self.assertEqual( "num_points_MUST_BE_INTEGER_IN_[1,6]", str( context.exception ) )
        with self.assertRaises( Exception ) as context:
            getNewtonCotesQuadrature( num_points = 7 )
        self.assertEqual( "num_points_MUST_BE_INTEGER_IN_[1,6]", str( context.exception ) )

    def test_return_types( self ):
        for num_points in range( 1, 7 ):
            x, w = getNewtonCotesQuadrature( num_points = num_points )
            self.assertIsInstance( obj = x, cls = numpy.ndarray )
            self.assertIsInstance( obj = w, cls = numpy.ndarray )
            self.assertTrue( len( x ) == num_points )
            self.assertTrue( len( w ) == num_points )