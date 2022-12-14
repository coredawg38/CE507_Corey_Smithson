import numpy
import unittest
import scipy
import basis ## USER DEFINED MODULE
import mesh ## USER DEFINED MODULE
import meshAproximationGalerkin as meshAG

class Test_assembleGramMatrix( unittest.TestCase ):
    def test_linear_lagrange( self ):
        domain = [ 0, 1 ]
        degree = [ 1, 1 ]
        node_coords, ien_array = mesh.generateMeshNonUniformDegree( domain[0], domain[1], degree )
        test_gram_matrix = meshAG.assembleGramMatrix( node_coords = node_coords, ien_array = ien_array, solution_basis = basis.evalLagrangeBasis1DanyDomain )
        gold_gram_matrix = numpy.array( [ [1/6, 1/12, 0], [1/12, 1/3, 1/12], [0, 1/12, 1/6] ] )
        self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )

    def test_quadratic_lagrange( self ):
        domain = [ 0, 1 ]
        degree = [ 2, 2 ]
        node_coords, ien_array = mesh.generateMeshNonUniformDegree( domain[0], domain[1], degree )
        test_gram_matrix = meshAG.assembleGramMatrix( node_coords = node_coords, ien_array = ien_array, solution_basis = basis.evalLagrangeBasis1DanyDomain )
        gold_gram_matrix = numpy.array( [ [1/15, 1/30, -1/60, 0, 0 ], [1/30, 4/15, 1/30, 0, 0], [-1/60, 1/30, 2/15, 1/30, -1/60], [ 0, 0, 1/30, 4/15, 1/30], [0, 0, -1/60, 1/30, 1/15] ] )
        self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )
    
    def test_cubic_lagrange( self ):
        domain = [ 0, 1 ]
        degree = [ 3, 3 ]
        node_coords, ien_array = mesh.generateMeshNonUniformDegree( domain[0], domain[1], degree )
        test_gram_matrix = meshAG.assembleGramMatrix( node_coords = node_coords, ien_array = ien_array, solution_basis = basis.evalLagrangeBasis1DanyDomain )
        gold_gram_matrix = numpy.array( [ [ 0.03809524,  0.02946429, -0.01071429,  0.00565476,  0.00000000,  0.00000000,  0.00000000 ], 
                                          [ 0.02946429,  0.19285714, -0.02410714, -0.01071429,  0.00000000,  0.00000000,  0.00000000 ], 
                                          [-0.01071429, -0.02410714,  0.19285714,  0.02946429,  0.00000000,  0.00000000,  0.00000000 ], 
                                          [ 0.00565476, -0.01071429,  0.02946429,  0.07619048,  0.02946429, -0.01071429,  0.00565476 ], 
                                          [ 0.00000000,  0.00000000,  0.00000000,  0.02946429,  0.19285714, -0.02410714, -0.01071429 ], 
                                          [ 0.00000000,  0.00000000,  0.00000000, -0.01071429, -0.02410714,  0.19285714,  0.02946429 ], 
                                          [ 0.00000000,  0.00000000,  0.00000000,  0.00565476, -0.01071429,  0.02946429,  0.03809524 ] ] )
        self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )
    
    def test_linear_bernstein( self ):
        domain = [ 0, 1 ]
        degree = [ 1, 1 ]
        node_coords, ien_array = mesh.generateMeshNonUniformDegree( domain[0], domain[1], degree )
        test_gram_matrix = meshAG.assembleGramMatrix( node_coords = node_coords, ien_array = ien_array, solution_basis = basis.evalBernsteinBasis1DanyDomain )
        gold_gram_matrix = numpy.array( [ [1/6, 1/12, 0], [1/12, 1/3, 1/12], [0, 1/12, 1/6] ] )
        self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )

    def test_quadratic_bernstein( self ):
        domain = [ 0, 1 ]
        degree = [ 2, 2 ]
        node_coords, ien_array = mesh.generateMeshNonUniformDegree( domain[0], domain[1], degree )
        test_gram_matrix = meshAG.assembleGramMatrix( node_coords = node_coords, ien_array = ien_array, solution_basis = basis.evalBernsteinBasis1DanyDomain )
        gold_gram_matrix = numpy.array( [ [1/10, 1/20, 1/60, 0, 0 ], [1/20, 1/15, 1/20, 0, 0 ], [1/60, 1/20, 1/5, 1/20, 1/60], [0, 0, 1/20, 1/15, 1/20], [0, 0, 1/60, 1/20, 1/10] ] )
        self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )
    
    def test_cubic_bernstein( self ):
        domain = [ 0, 1 ]
        degree = [ 3, 3 ]
        node_coords, ien_array = mesh.generateMeshNonUniformDegree( domain[0], domain[1], degree )
        test_gram_matrix = meshAG.assembleGramMatrix( node_coords = node_coords, ien_array = ien_array, solution_basis = basis.evalBernsteinBasis1DanyDomain )
        gold_gram_matrix = numpy.array( [ [1/14, 1/28, 1/70, 1/280, 0, 0, 0 ], [1/28, 3/70, 9/280, 1/70, 0, 0, 0 ], [1/70, 9/280, 3/70, 1/28, 0, 0, 0 ], [1/280, 1/70, 1/28, 1/7, 1/28, 1/70, 1/280], [0, 0, 0, 1/28, 3/70, 9/280, 1/70], [0, 0, 0, 1/70, 9/280, 3/70, 1/28], [0, 0, 0, 1/280, 1/70, 1/28, 1/14 ] ] )
        self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )

class Test_assembleForceVector( unittest.TestCase ):
    def test_lagrange_const_force_fun( self ):
        domain = [ 0, 1 ]
        degree = [ 3, 3 ]
        target_fun = lambda x: numpy.pi
        node_coords, ien_array = mesh.generateMeshNonUniformDegree( domain[0], domain[1], degree )
        test_force_vector = meshAG.assembleForceVector( target_fun = target_fun, node_coords = node_coords, ien_array = ien_array, solution_basis = basis.evalLagrangeBasis1DanyDomain )
        gold_force_vector = numpy.array( [ numpy.pi / 16.0, 3.0 * numpy.pi / 16.0, 3.0 * numpy.pi / 16.0, numpy.pi / 8.0, 3.0 * numpy.pi / 16.0, 3.0 * numpy.pi / 16.0, numpy.pi / 16.0 ] )
        self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )
    
    def test_lagrange_linear_force_fun( self ):
        domain = [ 0, 1 ]
        degree = [ 3, 3 ]
        target_fun = lambda x: 2*x + numpy.pi
        node_coords, ien_array = mesh.generateMeshNonUniformDegree( domain[0], domain[1], degree )
        test_force_vector = meshAG.assembleForceVector( target_fun = target_fun, node_coords = node_coords, ien_array = ien_array, solution_basis = basis.evalLagrangeBasis1DanyDomain )
        gold_force_vector = numpy.array( [ 0.20468287, 0.62654862, 0.73904862, 0.51769908, 0.81404862, 0.92654862, 0.31301621 ] )
        self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )
    
    def test_lagrange_quadratic_force_fun( self ):
        domain = [ 0, 1 ]
        degree = [ 3, 3 ]
        target_fun = lambda x: x**2.0
        node_coords, ien_array = mesh.generateMeshNonUniformDegree( domain[0], domain[1], degree )
        test_force_vector = meshAG.assembleForceVector( target_fun = target_fun, node_coords = node_coords, ien_array = ien_array, solution_basis = basis.evalLagrangeBasis1DanyDomain )
        gold_force_vector = numpy.array( [ 1.04166667e-03, 0, 2.81250000e-02, 3.33333333e-02, 6.56250000e-02, 1.50000000e-01, 5.52083333e-02 ] )
        self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )

    def test_lagrange_const_force_fun( self ):
        domain = [ 0, 1 ]
        degree = [ 3, 3 ]
        target_fun = lambda x: numpy.pi
        node_coords, ien_array = mesh.generateMeshNonUniformDegree( domain[0], domain[1], degree )
        test_force_vector = meshAG.assembleForceVector( target_fun = target_fun, node_coords = node_coords, ien_array = ien_array, solution_basis = basis.evalBernsteinBasis1DanyDomain )
        gold_force_vector = numpy.array( [ numpy.pi / 8.0, numpy.pi / 8.0, numpy.pi / 8.0, numpy.pi / 4.0, numpy.pi / 8.0, numpy.pi / 8.0, numpy.pi / 8.0 ] )
        self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )
    
    def test_bernstein_linear_force_fun( self ):
        domain = [ 0, 1 ]
        degree = [ 3, 3 ]
        target_fun = lambda x: 2*x + numpy.pi
        node_coords, ien_array = mesh.generateMeshNonUniformDegree( domain[0], domain[1], degree )
        test_force_vector = meshAG.assembleForceVector( target_fun = target_fun, node_coords = node_coords, ien_array = ien_array, solution_basis = basis.evalBernsteinBasis1DanyDomain )
        gold_force_vector = numpy.array( [ 0.41769908, 0.44269908, 0.46769908, 1.03539816, 0.56769908, 0.59269908, 0.61769908 ] )
        self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )
    
    def test_bernstein_quadratic_force_fun( self ):
        domain = [ 0, 1 ]
        degree = [ 3, 3 ]
        target_fun = lambda x: x**2.0
        node_coords, ien_array = mesh.generateMeshNonUniformDegree( domain[0], domain[1], degree )
        test_force_vector = meshAG.assembleForceVector( target_fun = target_fun, node_coords = node_coords, ien_array = ien_array, solution_basis = basis.evalBernsteinBasis1DanyDomain )
        gold_force_vector = numpy.array( [ 1/480, 1/160, 1/80, 1/15, 1/16, 13/160, 49/480 ] )
        self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )

class Test_computeSolution( unittest.TestCase ):
    def test_cubic_polynomial_target( self ):
        # print( "POLY TEST" )
        target_fun = lambda x: x**3 - (8/5)*x**2 + (3/5)*x
        domain = [ 0, 1 ]
        degree = [2]*2
        solution_basis = basis.evalBernsteinBasis1DanyDomain
        test_sol_coeff, node_coords, ien_array = meshAG.computeSolution( target_fun = target_fun, domain = domain, degree = degree, solution_basis =solution_basis )
        gold_sol_coeff = numpy.array( [ 1.0 / 120.0, 9.0 / 80.0, 1.0 / 40.0, -1.0 / 16.0, -1.0 / 120.0 ] )
        abs_err, rel_err = meshAG.computeFitError( target_fun, test_sol_coeff, node_coords, ien_array, solution_basis )
        # plotCompareGoldTestSolution( gold_sol_coeff, test_sol_coeff, node_coords, ien_array, solution_basis )
        # plotCompareFunToTestSolution( target_fun, test_sol_coeff, node_coords, ien_array, solution_basis )
        self.assertTrue( numpy.allclose( gold_sol_coeff, test_sol_coeff ) )
        self.assertAlmostEqual( first = rel_err, second = 0, delta = 1e-1 )

    def test_sin_target( self ):
        # print( "SIN TEST" )
        target_fun = lambda x: numpy.sin( numpy.pi * x )
        domain = [ 0, 1 ]
        degree = [2]*2
        solution_basis = basis.evalBernsteinBasis1DanyDomain
        test_sol_coeff, node_coords, ien_array = meshAG.computeSolution( target_fun = target_fun, domain = domain, degree = degree, solution_basis = solution_basis )
        gold_sol_coeff = numpy.array( [ -0.02607008, 0.9185523, 1.01739261, 0.9185523, -0.02607008 ] )
        abs_err, rel_err = meshAG.computeFitError( target_fun, test_sol_coeff, node_coords, ien_array, solution_basis )
        # plotCompareGoldTestSolution( gold_sol_coeff, test_sol_coeff, node_coords, ien_array, solution_basis )
        # plotCompareFunToTestSolution( target_fun, test_sol_coeff, node_coords, ien_array, solution_basis )
        self.assertAlmostEqual( first = rel_err, second = 0, delta = 1e-1 )
        
    def test_erfc_target( self ):
        # print( "ERFC TEST" )
        target_fun = lambda x: numpy.real( scipy.special.erfc( x ) )
        domain = [ -2, 2 ]
        degree = [3]*2
        solution_basis = basis.evalBernsteinBasis1DanyDomain
        test_sol_coeff, node_coords, ien_array = meshAG.computeSolution( target_fun = target_fun, domain = domain, degree = degree, solution_basis = solution_basis )
        gold_sol_coeff = numpy.array( [ 1.98344387, 2.0330054, 1.86372084, 1., 0.13627916, -0.0330054, 0.01655613 ] )
        abs_err, rel_err = meshAG.computeFitError( target_fun, test_sol_coeff, node_coords, ien_array, solution_basis )
        # plotCompareGoldTestSolution( gold_sol_coeff, test_sol_coeff, node_coords, ien_array, solution_basis )
        # plotCompareFunToTestSolution( target_fun, test_sol_coeff, node_coords, ien_array, solution_basis )
        self.assertAlmostEqual( first = rel_err, second = 0, delta = 1e-2 )
    
    def test_exptx_target( self ):
        # print( "EXPT TEST" )
        target_fun = lambda x: float( numpy.real( float( x )**float( x ) ) )
        domain = [ -1, 1 ]
        degree = [5]*2
        solution_basis = basis.evalBernsteinBasis1DanyDomain
        test_sol_coeff, node_coords, ien_array = meshAG.computeSolution( target_fun = target_fun, domain = domain, degree = degree, solution_basis = solution_basis )
        gold_sol_coeff = ( [ -1.00022471, -1.19005562, -0.9792369, 0.70884334, 1.73001439, 0.99212064, 0.44183573, 0.87014465, 0.5572111, 0.85241908, 0.99175228 ] )
        abs_err, rel_err = meshAG.computeFitError( target_fun, test_sol_coeff, node_coords, ien_array, solution_basis )
        # plotCompareGoldTestSolution( gold_sol_coeff, test_sol_coeff, node_coords, ien_array, solution_basis )
        # plotCompareFunToTestSolution( target_fun, test_sol_coeff, node_coords, ien_array, solution_basis )
        self.assertAlmostEqual( first = rel_err, second = 0, delta = 1e-2 )