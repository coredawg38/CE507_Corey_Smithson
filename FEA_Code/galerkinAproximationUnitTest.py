import basis
import unittest
import numpy
import galerkinAproximation
import scipy

class Test_computeSolution( unittest.TestCase ):
    def test_cubic_polynomial_target( self ):
        # print( "POLY TEST" )
        target_fun = lambda x: x**3 - (8/5)*x**2 + (3/5)*x
        domain = [ 0, 1 ]
        degree = 2
        solution_basis = basis.evalBernsteinBasis1DanyDomain
        test_sol_coeff = galerkinAproximation.computeGalerkinAproximation(func  = target_fun, domain = domain, degree = degree, solutionBasis = solution_basis )
        gold_sol_coeff = numpy.array( [ 1.0 / 20.0, 1.0 / 20.0, -1.0 / 20.0 ] )
        fit_err = galerkinAproximation.computeFitError( gold_sol_coeff, test_sol_coeff, domain, solution_basis )
        # plotCompareGoldTestSolution( gold_sol_coeff, test_sol_coeff, domain, solution_basis )
        # plotCompareFunToTestSolution( target_fun, test_sol_coeff, domain, solution_basis )
        self.assertTrue( numpy.allclose( gold_sol_coeff, test_sol_coeff ) )
        self.assertAlmostEqual( first = fit_err, second = 0, delta = 1e-12 )

    def test_sin_target( self ):
        # print( "SIN TEST" )
        target_fun = lambda x: numpy.sin( numpy.pi * x )
        domain = [ 0, 1 ]
        degree = 2
        solution_basis = basis.evalBernsteinBasis1DanyDomain
        test_sol_coeff = galerkinAproximation.computeGalerkinAproximation(func = target_fun, domain = domain, degree = degree, solutionBasis = solution_basis )
        gold_sol_coeff = numpy.array( [ (12*(numpy.pi**2 - 10))/(numpy.pi**3), -(6*(3*numpy.pi**2 - 40))/(numpy.pi**3), (12*(numpy.pi**2 - 10))/(numpy.pi**3)] )
        fit_err = galerkinAproximation.computeFitError( gold_sol_coeff, test_sol_coeff, domain, solution_basis )
        # plotCompareGoldTestSolution( gold_sol_coeff, test_sol_coeff, [0, 1], solution_basis )
        # plotCompareFunToTestSolution( target_fun, test_sol_coeff, domain, solution_basis )
        self.assertAlmostEqual( first = fit_err, second = 0, delta = 1e-5 )
        
    def test_erfc_target( self ):
        # print( "ERFC TEST" )
        target_fun = lambda x: numpy.real( scipy.special.erfc( x ) )
        domain = [ -2, 2 ]
        degree = 3
        solution_basis = basis.evalBernsteinBasis1DanyDomain
        test_sol_coeff = galerkinAproximation.computeGalerkinAproximation(func = target_fun, domain = domain, degree = degree, solutionBasis = solution_basis )
        gold_sol_coeff = numpy.array( [ 1.8962208131568558391841630949727, 2.6917062016799657617278998883219, -0.69170620167996576172789988832194, 0.10377918684314416081583690502732] )
        fit_err = galerkinAproximation.computeFitError( gold_sol_coeff, test_sol_coeff, domain, solution_basis )
        # plotCompareGoldTestSolution( gold_sol_coeff, test_sol_coeff, [-2, 2], solution_basis )
        # plotCompareFunToTestSolution( target_fun, test_sol_coeff, domain, solution_basis )
        self.assertAlmostEqual( first = fit_err, second = 0, delta = 1e-4 )
    
    def test_exptx_target( self ):
        # print( "EXPT TEST" )
        target_fun = lambda x: float( numpy.real( float( x )**float( x ) ) )
        domain = [ -1, 1 ]
        degree = 5
        solution_basis = basis.evalBernsteinBasis1DanyDomain
        test_sol_coeff = galerkinAproximation.computeGalerkinAproximation(func = target_fun, domain = domain, degree = degree, solutionBasis = solution_basis )
        gold_sol_coeff = ( [ -0.74841381974620419634327921170757, -3.4222814978197825394922980704166, 7.1463655364038831935841354617843, -2.9824200396151998304868767455064, 1.6115460899636204992283970407553, 0.87876479932866366847320748048494 ] )
        fit_err = galerkinAproximation.computeFitError( gold_sol_coeff, test_sol_coeff, domain, solution_basis )
        # plotCompareGoldTestSolution( gold_sol_coeff, test_sol_coeff, [-1, 1], solution_basis )
        # plotCompareFunToTestSolution( target_fun, test_sol_coeff, domain, solution_basis )
        self.assertAlmostEqual( first = fit_err, second = 0, delta = 1e-2 )

class Test_assembleForceVector( unittest.TestCase ):
    def test_legendre_const_force_fun( self ):
        test_force_vector = galerkinAproximation.assembleForceVector( func = lambda x: numpy.pi, domain = [0, 1], degree = 1, solutionBasis = basis.evalLegendreBasis1DanyDomain )
        gold_force_vector = numpy.array( [ numpy.pi, 0.0 ] )
        self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )
    
    def test_legendre_linear_force_fun( self ):
        test_force_vector = galerkinAproximation.assembleForceVector( func = lambda x: 2*x + numpy.pi, domain = [0, 1], degree = 1, solutionBasis = basis.evalLegendreBasis1DanyDomain )
        gold_force_vector = numpy.array( [ numpy.pi + 1.0, 1.0/3.0 ] )
        self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )
    
    def test_legendre_quadratic_force_fun( self ):
        test_force_vector = galerkinAproximation.assembleForceVector( func = lambda x: x**2.0, domain = [0, 1], degree = 1, solutionBasis = basis.evalLegendreBasis1DanyDomain )
        gold_force_vector = numpy.array( [ 1.0/3.0, 1.0/6.0 ] )
        self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )
        test_force_vector = galerkinAproximation.assembleForceVector( func = lambda x: x**2.0, domain = [0, 1], degree = 2, solutionBasis = basis.evalLegendreBasis1DanyDomain )
        gold_force_vector = numpy.array( [ 1.0/3.0, 1.0/6.0, 1.0/30.0 ] )
        self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )

    def test_lagrange_const_force_fun( self ):
        test_force_vector = galerkinAproximation.assembleForceVector( func = lambda x: numpy.pi, domain = [0, 1], degree = 1, solutionBasis = basis.evalLagrangeBasis1DanyDomain )
        gold_force_vector = numpy.array( [ numpy.pi / 2.0, numpy.pi / 2.0 ] )
        self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )
    
    def test_lagrange_linear_force_fun( self ):
        test_force_vector = galerkinAproximation.assembleForceVector( func = lambda x: 2*x + numpy.pi, domain = [0, 1], degree = 1, solutionBasis = basis.evalLagrangeBasis1DanyDomain )
        gold_force_vector = numpy.array( [ numpy.pi/2.0 + 1.0/3.0, numpy.pi/2.0 + 2.0/3.0 ] )
        self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )
    
    def test_lagrange_quadratic_force_fun( self ):
        test_force_vector = galerkinAproximation.assembleForceVector( func = lambda x: x**2.0, domain = [0, 1], degree = 1, solutionBasis = basis.evalLagrangeBasis1DanyDomain )
        gold_force_vector = numpy.array( [ 1.0/12.0, 1.0/4.0 ] )
        self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )
        test_force_vector = galerkinAproximation.assembleForceVector( func = lambda x: x**2.0, domain = [0, 1], degree = 2, solutionBasis = basis.evalLagrangeBasis1DanyDomain )
        gold_force_vector = numpy.array( [ -1.0/60.0, 1.0/5.0, 3.0/20.0 ] )
        self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )

    def test_bernstein_const_force_fun( self ):
        test_force_vector = galerkinAproximation.assembleForceVector( func = lambda x: numpy.pi, domain = [0, 1], degree = 1, solutionBasis = basis.evalBernsteinBasis1DanyDomain )
        gold_force_vector = numpy.array( [ numpy.pi / 2.0, numpy.pi / 2.0 ] )
        self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )
    
    def test_bernstein_linear_force_fun( self ):
        test_force_vector = galerkinAproximation.assembleForceVector( func = lambda x: 2*x + numpy.pi, domain = [0, 1], degree = 1, solutionBasis = basis.evalBernsteinBasis1DanyDomain )
        gold_force_vector = numpy.array( [ numpy.pi/2.0 + 1.0/3.0, numpy.pi/2.0 + 2.0/3.0 ] )
        self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )
    
    def test_bernstein_quadratic_force_fun( self ):
        test_force_vector = galerkinAproximation.assembleForceVector( func = lambda x: x**2.0, domain = [0, 1], degree = 1, solutionBasis = basis.evalBernsteinBasis1DanyDomain )
        gold_force_vector = numpy.array( [ 1.0/12.0, 1.0/4.0 ] )
        self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )
        test_force_vector = galerkinAproximation.assembleForceVector( func = lambda x: x**2.0, domain = [0, 1], degree = 2, solutionBasis = basis.evalBernsteinBasis1DanyDomain )
        gold_force_vector = numpy.array( [ 1.0/30.0, 1.0/10.0, 1.0/5.0 ] )
        self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )

class Test_assembleGramMatrix( unittest.TestCase ):
    def test_quadratic_legendre( self ):
        test_gram_matrix = galerkinAproximation.assembleGramMatrix( domain = [0, 1], degree = 2, solutionBasis = basis.evalLegendreBasis1DanyDomain )
        gold_gram_matrix = numpy.array( [ [1.0, 0.0, 0.0], [0.0, 1.0/3.0, 0.0], [0.0, 0.0, 0.2] ] )
        self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )
    
    def test_cubic_legendre( self ):
        test_gram_matrix = galerkinAproximation.assembleGramMatrix( domain = [0, 1], degree = 3, solutionBasis = basis.evalLegendreBasis1DanyDomain )
        gold_gram_matrix = numpy.array( [ [1.0, 0.0, 0.0, 0.0], [0.0, 1.0/3.0, 0.0, 0.0], [0.0, 0.0, 0.2, 0.0], [ 0.0, 0.0, 0.0, 1.0/7.0] ] )
        self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )

    def test_linear_bernstein( self ):
        test_gram_matrix = galerkinAproximation.assembleGramMatrix( domain = [0, 1], degree = 1, solutionBasis = basis.evalBernsteinBasis1DanyDomain )
        gold_gram_matrix = numpy.array( [ [1.0/3.0, 1.0/6.0], [1.0/6.0, 1.0/3.0] ] )
        self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )

    def test_quadratic_bernstein( self ):
        test_gram_matrix = galerkinAproximation.assembleGramMatrix( domain = [0, 1], degree = 2, solutionBasis = basis.evalBernsteinBasis1DanyDomain )
        gold_gram_matrix = numpy.array( [ [0.2, 0.1, 1.0/30.0], [0.1, 2.0/15.0, 0.1], [1.0/30.0, 0.1, 0.2] ] )
        self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )
    
    def test_cubic_bernstein( self ):
        test_gram_matrix = galerkinAproximation.assembleGramMatrix( domain = [0, 1], degree = 3, solutionBasis = basis.evalBernsteinBasis1DanyDomain )
        gold_gram_matrix = numpy.array( [ [1.0/7.0, 1.0/14.0, 1.0/35.0, 1.0/140.0], [1.0/14.0, 3.0/35.0, 9.0/140.0, 1.0/35.0], [1.0/35.0, 9.0/140.0, 3.0/35.0, 1.0/14.0], [ 1.0/140.0, 1.0/35.0, 1.0/14.0, 1.0/7.0] ] )
        self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )

    def test_linear_lagrange( self ):
        test_gram_matrix = galerkinAproximation.assembleGramMatrix( domain = [0, 1], degree = 1, solutionBasis = basis.evalLagrangeBasis1DanyDomain )
        gold_gram_matrix = numpy.array( [ [1.0/3.0, 1.0/6.0], [1.0/6.0, 1.0/3.0] ] )
        self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )

    def test_quadratic_lagrange( self ):
        test_gram_matrix = galerkinAproximation.assembleGramMatrix( domain = [0, 1], degree = 2, solutionBasis = basis.evalLagrangeBasis1DanyDomain )
        gold_gram_matrix = numpy.array( [ [2.0/15.0, 1.0/15.0, -1.0/30.0], [1.0/15.0, 8.0/15.0, 1.0/15.0], [-1.0/30.0, 1.0/15.0, 2.0/15.0] ] )
        self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )
    
    def test_cubic_lagrange( self ):
        test_gram_matrix = galerkinAproximation.assembleGramMatrix( domain = [0, 1], degree = 3, solutionBasis = basis.evalLagrangeBasis1DanyDomain )
        gold_gram_matrix = numpy.array( [ [8.0/105.0, 33.0/560.0, -3.0/140.0, 19.0/1680.0], [33.0/560.0, 27.0/70.0, -27.0/560.0, -3.0/140.0], [-3.0/140.0, -27.0/560.0, 27.0/70.0, 33/560.0], [ 19.0/1680.0, -3.0/140.0, 33.0/560.0, 8.0/105.0] ] )
        self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )