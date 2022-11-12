import numpy
import unittest


def generateMesh(xmin, xmax, num_elems, degree):

    elem_size = (xmax - xmin)/(num_elems*degree)
    node_coords = numpy.zeros( (num_elems*degree) + 1)
    a_shape = (num_elems, degree+1)  # 3 rows and 4 columns
    ien_array = numpy.zeros(a_shape, dtype=int)
    for i in range(0,num_elems*degree):
        if (i==0):
            node_coords[i]=xmin
        else:
            node_coords[i] = node_coords[i-1] + elem_size
    node_coords[num_elems*degree]=xmax

    counter = 0
    for i in range(0, num_elems):
        for j in range(0,degree+1):
            ien_array[i,j] = counter
            counter = counter + 1
        counter = counter - 1

    return node_coords, ien_array

def generateMeshNonUniformDegree(xmin, xmax, degree):

    #num_nodes = 1
    num_elems = len(degree)
    elem_size = (xmax - xmin)/(num_elems)
    node_coords = numpy.array([xmin])
    ien_array = {0: [0]}
    counter = 0
    for i in range(0, num_elems):
        element_nodes = []
        for j in range(0,degree[i]):
            last_coord = node_coords[-1]
            dist = elem_size/degree[i]
            node_coords = numpy.append(node_coords, [last_coord+dist])

            element_nodes.append(counter)
            counter += 1
        element_nodes.append(counter)
        ien_array[i] = element_nodes
        #counter = counter - 1

    return node_coords, ien_array

def getElementNodes( ien_array, elem_idx ):
    return ien_array[elem_idx]

def getElementIdxContainingPoint(x,node_coords,ien_array):
    num_elems = len( ien_array )
    for elem_idx in range (0, num_elems):
        elem_nodes = ien_array[elem_idx]
        elem_domain = numpy.array( [ node_coords[ elem_nodes[0] ], node_coords[ elem_nodes[-1] ] ] )
        if(x >=  elem_domain[0] and x <= elem_domain[1]):
            return elem_idx
    raise Exception( "ELEMENT_CONTAINING_POINT_NOT_FOUND" )

def getElemDomain(node_coords, ien_array, elem_idx ):
    elem_nodes = ien_array[elem_idx]
    elem_domain = [node_coords[elem_nodes[0]], node_coords[elem_nodes[-1]]]
    return elem_domain

def getGlobalNodeID(ien_array,elem_idx,a):
    return ien_array[elem_idx][a]



