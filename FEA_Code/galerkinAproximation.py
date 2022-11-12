import numpy
import basis
import unittest
import quadrature
import scipy
import matplotlib.pyplot as plt

def assembleGramMatrix(domain, degree, solutionBasis):
    M = numpy.zeros([degree+1,degree+1])
    numPoints = int( numpy.ceil( ( 2*degree + 1 ) / 2.0 ) )
    for i in range(0,degree+1):
        N_i= lambda x:  solutionBasis(x,degree,i,[-1,1])
        for j in range(0,degree+1):
            N_j= lambda x: solutionBasis(x,degree,j,[-1,1])
            integrand = lambda x: N_i(x) * N_j(x)
            M[i,j] = quadrature.quad( integrand, domain, numPoints )
    return M

def assembleForceVector(func, domain, degree, solutionBasis):
    F = numpy.zeros([degree+1])
    numPoints = 4
    for i in range(0,degree+1):
            N_i= lambda x:  solutionBasis(x,degree,i,[-1,1])
            integrand = lambda x: N_i(x)* func(basis.affineMapping1D( [-1, 1], domain, x) )
            F[i] = quadrature.quad( integrand, domain, numPoints )
    return F

def computeGalerkinAproximation(func, domain, degree, solutionBasis):
    M = assembleGramMatrix(domain, degree, solutionBasis)
    F = assembleForceVector(func, domain, degree, solutionBasis)
    d = numpy.matmul(numpy.linalg.inv(M),F)
    return d

def evaluateSolutionAt( x, domain, coeff, solution_basis ):
    degree = len( coeff ) - 1
    y = 0.0
    for n in range( 0, len( coeff ) ):
        y += coeff[n] * solution_basis( variate = x, degree = degree, basis_idx = n, domain = domain)
    return y

def computeFitError( gold_coeff, test_coeff, domain, solution_basis ):
    err_fun = lambda x: abs( evaluateSolutionAt( x, domain, gold_coeff, solution_basis ) - evaluateSolutionAt( x, domain, test_coeff, solution_basis ) )
    abs_err, _ = scipy.integrate.quad( err_fun, domain[0], domain[1], epsrel = 1e-12, limit = 1000 )
    return abs_err

def plotCompareGoldTestSolution( gold_coeff, test_coeff, domain, solution_basis ):
    x = numpy.linspace( domain[0], domain[1], 1000 )
    yg = numpy.zeros( 1000 )
    yt = numpy.zeros( 1000 )
    for i in range(0, len(x) ):
        yg[i] = evaluateSolutionAt( x[i], domain, gold_coeff, solution_basis )
        yt[i] = evaluateSolutionAt( x[i], domain, test_coeff, solution_basis )
    plt.plot( x, yg )
    plt.plot( x, yt )
    plt.show()

def plotCompareFunToTestSolution( fun, test_coeff, domain, solution_basis ):
    x = numpy.linspace( domain[0], domain[1], 1000 )
    y = numpy.zeros( 1000 )
    yt = numpy.zeros( 1000 )
    for i in range(0, len(x) ):
        y[i] = fun( x[i] )
        yt[i] = evaluateSolutionAt( x[i], domain, test_coeff, solution_basis )
    plt.plot( x, y )
    plt.plot( x, yt )
    plt.show()


# target_fun = lambda x: x**3 - (8/5)*x**2 + (3/5)*x
# domain = [ 0, 1 ]
# degree = 2
# solution_basis = basis.evalLagrangeBasis1DanyDomain
# variate = lambda x: x

# M = assembleGramMatrix(domain, degree, solution_basis)
# print(M)
# F = assembleForceVector(target_fun, domain, degree, solution_basis)
# print(F)
# d = computeGalerkinAproximation(target_fun, domain, degree, solution_basis)
# print(d)

# plotCompareFunToTestSolution( target_fun, [0,8/135,-2/135,0], domain, solution_basis )
# plotCompareFunToTestSolution( target_fun, d, domain, solution_basis )
# solution_basis = basis.evalBernsteinBasis1DanyDomain
# variate = lambda x: x
# d = computeGalerkinAproximation(target_fun, domain, degree, solution_basis)
# print(d)
# plotCompareFunToTestSolution( target_fun, d, domain, solution_basis )
