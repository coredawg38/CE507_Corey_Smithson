import unittest
import numpy
import math

def affineMapping1D( domain, target_domain, x ):
    A = numpy.array( [ [ 1.0, domain[0] ], [ 1.0, domain[1] ] ] )
    b = numpy.array( [target_domain[0], target_domain[1] ] )
    c = numpy.linalg.solve( A, b )
    fx = c[0] + c[1] * x
    return fx

def evalLagrangeBasis1DanyDomain(variate, degree, basis_idx, domain):
    variate = affineMapping1D( domain, [-1, 1], variate )  

    nodes = numpy.linspace(-1,1,degree+1)
    value = 1  

    for i in range(0,degree+1):
        if(i != basis_idx):
            value *=  (variate - nodes[i] )/(nodes[basis_idx] - nodes[i])

    return value

def evalLegendreBasis1DanyDomain( variate, degree, basis_idx, domain):
    variate = affineMapping1D( domain, [-1, 1], variate )
    if basis_idx == 0:
        value = 1.0
    elif basis_idx == 1:
        value = variate
    else:
        i = basis_idx - 1
        term1 = i * evalLegendreBasis1DanyDomain(variate,degree,i-1,[-1,1])
        term2 = ((2*i)+1) * variate * evalLegendreBasis1DanyDomain(variate, degree, i, [-1,1])
        value = (term2 - term1) / (i +1)
    return value

def evalBernsteinBasis1DanyDomain(variate, degree, basis_idx, domain):
    variate = affineMapping1D( domain, [0, 1], variate )
    coefficient = math.comb(degree, basis_idx)
    value = coefficient * variate**(basis_idx) * (1.0- variate)**(degree - basis_idx)

    return value



def evalLegendreBasis1D( degree, variate):
    if degree == 0:
        value = 1.0
    elif degree == 1:
        value = variate
    else:
        i = degree - 1
        term1 = i * evalLegendreBasis1D(degree = i-1, variate = variate)
        term2 = ((2*i)+1) * variate * evalLegendreBasis1D(degree = i, variate = variate)
        value = (term2 - term1) / (i +1)
    return value

def evalLagrangeBasis1D(variate, degree, basis_idx):
        
    jVec = numpy.linspace(0,degree,num=degree+1,dtype="int")
    jVec = numpy.delete(jVec,basis_idx)
    nodes = numpy.linspace(-1,1,num=degree+1)
        
    value = 1

    for i in jVec:
        value *=  (variate - nodes[i] )/(nodes[basis_idx] - nodes[i])

    return value

def evalBernsteinBasis1D(variate, degree, basis_idx):
    variate = (variate +1)/2
    coefficient = math.factorial(degree)/(math.factorial(basis_idx) * math.factorial(degree-basis_idx))
    value = coefficient * variate**(basis_idx) * (1- variate)**(degree - basis_idx)

    return value

