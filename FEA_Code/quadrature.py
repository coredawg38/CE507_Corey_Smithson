import numpy
import scipy
import basis
import math
import MomentFit

def quad( fun, domain, num_points ):
    jacobian = ( domain[1] - domain[0] ) / ( 1 - (-1) )
    x_qp, w_qp = getGaussLegendreQuadrature( num_points)
    integral = 0.0
    for qp in range( 0, len( x_qp ) ):
        integral += ( fun( x_qp[qp] ) * w_qp[qp] ) * jacobian
    return integral

def getGaussLegendreQuadrature( numPoints):
    if numPoints == 1:
        x = [ 0.0 ]
        w = [ 2.0 ]
    elif numPoints == 2:
        x = [ -1.0 / math.sqrt(3), 
              +1.0 / math.sqrt(3) ]

        w = [ 1.0, 
              1.0  ]
    elif numPoints == 3:
        x = [ -1.0 * math.sqrt( 3.0 / 5.0 ), 
               0.0, 
              +1.0 * math.sqrt( 3.0 / 5.0 ) ]

        w = [ 5.0 / 9.0, 
              8.0 / 9.0, 
              5.0 / 9.0 ]
    elif numPoints == 4:
        x = [ -1.0 * math.sqrt( 3.0 / 7.0 + 2.0 / 7.0 * math.sqrt( 6.0 / 5.0 ) ),
              -1.0 * math.sqrt( 3.0 / 7.0 - 2.0 / 7.0 * math.sqrt( 6.0 / 5.0 ) ),
              +1.0 * math.sqrt( 3.0 / 7.0 - 2.0 / 7.0 * math.sqrt( 6.0 / 5.0 ) ),
              +1.0 * math.sqrt( 3.0 / 7.0 + 2.0 / 7.0 * math.sqrt( 6.0 / 5.0 ) ) ]
        
        w = [ ( 18.0 - math.sqrt( 30.0 ) ) / 36.0,
              ( 18.0 + math.sqrt( 30.0 ) ) / 36.0,
              ( 18.0 + math.sqrt( 30.0 ) ) / 36.0,
              ( 18.0 - math.sqrt( 30.0 ) ) / 36.0 ]
    elif numPoints == 5:
        x = [ -1.0 / 3.0 * math.sqrt( 5.0 + 2.0 * math.sqrt( 10.0 / 7.0 ) ),
              -1.0 / 3.0 * math.sqrt( 5.0 - 2.0 * math.sqrt( 10.0 / 7.0 ) ),
               0.0,
              +1.0 / 3.0 * math.sqrt( 5.0 - 2.0 * math.sqrt( 10.0 / 7.0 ) ),
              +1.0 / 3.0 * math.sqrt( 5.0 + 2.0 * math.sqrt( 10.0 / 7.0 ) ) ]
        
        w = [ ( 322.0 - 13.0 * math.sqrt( 70.0 ) ) / 900.0,
              ( 322.0 + 13.0 * math.sqrt( 70.0 ) ) / 900.0,
                128.0 / 225.0,
              ( 322.0 + 13.0 * math.sqrt( 70.0 ) ) / 900.0,
              ( 322.0 - 13.0 * math.sqrt( 70.0 ) ) / 900.0, ]
    elif numPoints > 5:
        x,w = MomentFit.computeGaussLegendreQuadrature( numPoints )
    else:
        raise( Exception( "num_points_MUST_BE_POSITIVE_INTEGER" ) )
    return x, w

