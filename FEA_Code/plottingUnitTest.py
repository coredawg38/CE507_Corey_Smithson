import unittest
import matplotlib.pyplot as plt
import mesh
import numpy
import plotting
import basis
import scipy

class Test_plotBasisMesh( unittest.TestCase ):
    def test_3_linear_lagrange( self ):
        node_coords, ien_array = mesh.generateMeshNonUniformDegree( 0, 3, [ 1, 1, 1 ] )
        coeff = numpy.ones( shape = node_coords.shape[0] )
        plotting.plot_basis_functions( ien_array = ien_array, node_coords = node_coords, coeff = coeff, eval_basis = basis.evalLagrangeBasis1DanyDomain)
        plt.close()

    def test_3_quadratic_lagrange( self ):
        node_coords, ien_array = mesh.generateMeshNonUniformDegree( 0, 3, [ 2, 2, 2 ] )
        coeff = numpy.ones( shape = node_coords.shape[0] )
        plotting.plot_basis_functions( ien_array = ien_array, node_coords = node_coords, coeff = coeff, eval_basis = basis.evalLagrangeBasis1DanyDomain)
        plt.close()

    def test_3_quadratic_bernstein( self ):
        node_coords, ien_array = mesh.generateMeshNonUniformDegree( 0, 3, [ 2, 2, 2 ] )
        coeff = numpy.ones( shape = node_coords.shape[0] )
        plotting.plot_basis_functions( ien_array = ien_array, node_coords = node_coords, coeff = coeff, eval_basis = basis.evalBernsteinBasis1DanyDomain)
        plt.close()

    def test_4_p_refine_lagrange( self ):
        node_coords, ien_array = mesh.generateMeshNonUniformDegree( 0, 4, [ 1, 2, 3, 4 ] )
        coeff = numpy.ones( shape = node_coords.shape[0] )
        plotting.plot_basis_functions( ien_array = ien_array, node_coords = node_coords, coeff = coeff, eval_basis = basis.evalLagrangeBasis1DanyDomain)
        plt.close()

    def test_4_p_refine_bernstein( self ):
        node_coords, ien_array = mesh.generateMeshNonUniformDegree( 0, 4, [ 1, 2, 3, 4 ] )
        coeff = numpy.ones( shape = node_coords.shape[0] )
        plotting.plot_basis_functions( ien_array = ien_array, node_coords = node_coords, coeff = coeff, eval_basis = basis.evalBernsteinBasis1DanyDomain)
        plt.close()

    def test_10_linear_lagrange( self ):
        node_coords, ien_array = mesh.generateMeshNonUniformDegree( 0, 10, [ 1 ]*10 )
        coeff = numpy.ones( shape = node_coords.shape[0] )
        plotting.plot_basis_functions( ien_array = ien_array, node_coords = node_coords, coeff = coeff, eval_basis = basis.evalLagrangeBasis1DanyDomain)
        plt.close()

    def test_10_quadratic_lagrange( self ):
        node_coords, ien_array = mesh.generateMeshNonUniformDegree( 0, 10, [ 2 ]*10 )
        coeff = numpy.ones( shape = node_coords.shape[0] )
        plotting.plot_basis_functions( ien_array = ien_array, node_coords = node_coords, coeff = coeff, eval_basis = basis.evalLagrangeBasis1DanyDomain)
        plt.close()
    
    def test_approx_erfc_10_linear_lagrange( self ):
        node_coords, ien_array = mesh.generateMeshNonUniformDegree( -2, 2, [ 1 ]*10 )
        coeff = scipy.special.erfc( node_coords )
        plotting.plot_basis_functions( ien_array = ien_array, node_coords = node_coords, coeff = coeff, eval_basis = basis.evalLagrangeBasis1DanyDomain)
        plt.close()
    
    def test_approx_erfc_10_quadratic_lagrange( self ):
        node_coords, ien_array = mesh.generateMeshNonUniformDegree( -2, 2, [ 2 ]*10 )
        coeff = scipy.special.erfc( node_coords )
        plotting.plot_basis_functions( ien_array = ien_array, node_coords = node_coords, coeff = coeff, eval_basis = basis.evalLagrangeBasis1DanyDomain)
        plt.close()

class Test_plotPiecewiseApproximation( unittest.TestCase ):
    def test_approx_erfc_10_quadratic_lagrange( self ):
        node_coords, ien_array = mesh.generateMeshNonUniformDegree( -2, 2, [ 2, 2 ] )
        coeff = scipy.special.erfc( node_coords )
        plotting.plotPiecewiseApproximation( ien_array = ien_array, node_coords = node_coords, coeff = coeff, eval_basis = basis.evalLagrangeBasis1DanyDomain)
        plt.close()

    def test_approx_erfc_5_p_refine_lagrange( self ):
        node_coords, ien_array = mesh.generateMeshNonUniformDegree( -2, 2, [ 2, 1, 2 ] )
        coeff = scipy.special.erfc( node_coords )
        plotting.plotPiecewiseApproximation( ien_array = ien_array, node_coords = node_coords, coeff = coeff, eval_basis = basis.evalLagrangeBasis1DanyDomain)
        plt.close()



