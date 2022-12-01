import unittest
import uspline
import bext
import numpy
import splineBarGalerkin

class test_assembleStressMatrix( unittest.TestCase ):
       def test_one_linear_C0_element( self ):
              problem = { "elastic_modulus": 100,
                     "area": 0.01,
                     "length": 1.0,
                     "traction": { "value": 1e-3, "position": 1.0 },
                     "displacement": { "value": 0.0, "position": 0.0 },
                     "body_force": 0.0 }
              spline_space = { "domain": [0, problem[ "length" ]], "degree": [ 1 ], "continuity": [ -1, -1 ] }
              uspline.make_uspline_mesh( spline_space, "temp_uspline" )
              uspline_bext = bext.readBEXT( "temp_uspline.json" )
              test_stiffness_matrix = splineBarGalerkin.assembleStiffnessMatrix( problem = problem, uspline_bext = uspline_bext )
              gold_stiffness_matrix = numpy.array( [ [ 1.0, -1.0 ], [ -1.0, 1.0 ] ] )
              self.assertTrue( numpy.allclose( test_stiffness_matrix, gold_stiffness_matrix ) )

       def test_two_linear_C0_element( self ):
              problem = { "elastic_modulus": 100,
                     "area": 0.01,
                     "length": 1.0,
                     "traction": { "value": 1e-3, "position": 1.0 },
                     "displacement": { "value": 0.0, "position": 0.0 },
                     "body_force": 0.0 }
              spline_space = { "domain": [0, problem[ "length" ]], "degree": [ 1, 1 ], "continuity": [ -1, 0, -1 ] }
              uspline.make_uspline_mesh( spline_space, "temp_uspline" )
              uspline_bext = bext.readBEXT( "temp_uspline.json" )
              test_stiffness_matrix = splineBarGalerkin.assembleStiffnessMatrix( problem = problem, uspline_bext = uspline_bext )
              gold_stiffness_matrix = numpy.array( [ [ 2.0, -2.0, 0.0 ], [ -2.0, 4.0, -2.0 ], [ 0.0, -2.0, 2.0 ] ] )
              self.assertTrue( numpy.allclose( test_stiffness_matrix, gold_stiffness_matrix ) )

       def test_one_quadratic_C0_element( self ):
              problem = { "elastic_modulus": 100,
                     "area": 0.01,
                     "length": 1.0,
                     "traction": { "value": 1e-3, "position": 1.0 },
                     "displacement": { "value": 0.0, "position": 0.0 },
                     "body_force": 0.0 }
              spline_space = { "domain": [0, problem[ "length" ]], "degree": [ 2 ], "continuity": [ -1, -1 ] }
              uspline.make_uspline_mesh( spline_space, "temp_uspline" )
              uspline_bext = bext.readBEXT( "temp_uspline.json" )
              test_stiffness_matrix = splineBarGalerkin.assembleStiffnessMatrix( problem = problem, uspline_bext = uspline_bext )
              gold_stiffness_matrix = numpy.array( [ [  4.0 / 3.0, -2.0 / 3.0, -2.0 / 3.0 ],
                                                 [ -2.0 / 3.0,  4.0 / 3.0, -2.0 / 3.0 ],
                                                 [ -2.0 / 3.0, -2.0 / 3.0,  4.0 / 3.0 ] ] )
              self.assertTrue( numpy.allclose( test_stiffness_matrix, gold_stiffness_matrix ) )

       def test_two_quadratic_C0_element( self ):
              problem = { "elastic_modulus": 100,
                     "area": 0.01,
                     "length": 1.0,
                     "traction": { "value": 1e-3, "position": 1.0 },
                     "displacement": { "value": 0.0, "position": 0.0 },
                     "body_force": 0.0 }
              spline_space = { "domain": [0, problem[ "length" ]], "degree": [ 2, 2 ], "continuity": [ -1, 0, -1 ] }
              uspline.make_uspline_mesh( spline_space, "temp_uspline" )
              uspline_bext = bext.readBEXT( "temp_uspline.json" )
              test_stiffness_matrix = splineBarGalerkin.assembleStiffnessMatrix( problem = problem, uspline_bext = uspline_bext )
              gold_stiffness_matrix = numpy.array( [ [  8.0 / 3.0, -4.0 / 3.0, -4.0 / 3.0,  0.0,        0.0 ],
                                                 [ -4.0 / 3.0,  8.0 / 3.0, -4.0 / 3.0,  0.0,        0.0 ],
                                                 [ -4.0 / 3.0, -4.0 / 3.0, 16.0 / 3.0, -4.0 / 3.0, -4.0 / 3.0 ],
                                                 [  0.0,        0.0,       -4.0 / 3.0,  8.0 / 3.0, -4.0 / 3.0 ],
                                                 [  0.0,        0.0,       -4.0 / 3.0, -4.0 / 3.0,  8.0 / 3.0 ] ] )
              self.assertTrue( numpy.allclose( test_stiffness_matrix, gold_stiffness_matrix ) )

       def test_two_quadratic_C1_element( self ):
              problem = { "elastic_modulus": 100,
                     "area": 0.01,
                     "length": 1.0,
                     "traction": { "value": 1e-3, "position": 1.0 },
                     "displacement": { "value": 0.0, "position": 0.0 },
                     "body_force": 0.0 }
              spline_space = { "domain": [0, problem[ "length" ]], "degree": [ 2, 2 ], "continuity": [ -1, 1, -1 ] }
              uspline.make_uspline_mesh( spline_space, "temp_uspline" )
              uspline_bext = bext.readBEXT( "temp_uspline.json" )
              test_stiffness_matrix = splineBarGalerkin.assembleStiffnessMatrix( problem = problem, uspline_bext = uspline_bext )
              gold_stiffness_matrix = numpy.array( [ [  8.0 / 3.0, -2.0,       -2.0/ 3.0,   0.0 ],
                                                 [ -2.0,        8.0 / 3.0,  0.0,       -2.0 / 3.0 ],
                                                 [ -2.0 / 3.0,  0.0,        8.0 / 3.0, -2.0 ],
                                                 [  0.0,       -2.0 / 3.0, -2.0,        8.0 / 3.0 ] ] )
              self.assertTrue( numpy.allclose( test_stiffness_matrix, gold_stiffness_matrix ) )

class test_ComputeSolution( unittest.TestCase ):
    def test_simple( self ):
           problem = { "elastic_modulus": 100,
                       "area": 0.01,
                       "length": 1.0,
                       "traction": { "value": 1e-3, "position": 1.0 },
                       "displacement": { "value": 0.0, "position": 0.0 },
                       "body_force": 1e-3 }
           spline_space = { "domain": [0, problem[ "length" ]], "degree": [ 1, 1, 1 ], "continuity": [ -1, 0, 0, -1 ] }
           uspline.make_uspline_mesh( spline_space, "temp_uspline" )
           uspline_bext = bext.readBEXT( "temp_uspline.json" )
           test_sol_coeff = splineBarGalerkin.computeSolution( problem = problem, uspline_bext = uspline_bext )
           gold_sol_coeff = numpy.array( [ 0.0, 11.0 / 18000.0, 1.0 / 900.0, 3.0 / 2000.0 ] )
           self.assertTrue( numpy.allclose( test_sol_coeff, gold_sol_coeff ) )
           splineBarGalerkin.plotSolution( test_sol_coeff, uspline_bext )
           splineBarGalerkin.plotCompareFunToExactSolution( problem, test_sol_coeff, uspline_bext )

    def test_textbook_problem( self ):
           problem = { "elastic_modulus": 200e9,
                       "area": 1.0,
                       "length": 5.0,
                       "traction": { "value": 9810.0, "position": 5.0 },
                       "displacement": { "value": 0.0, "position": 0.0 },
                       "body_force": 784800.0 }
           spline_space = { "domain": [0, problem[ "length" ]], "degree": [ 2, 2 ], "continuity": [ -1, 1, -1 ] }
           uspline.make_uspline_mesh( spline_space, "temp_uspline" )
           uspline_bext = bext.readBEXT( "temp_uspline.json" )
           test_sol_coeff = splineBarGalerkin.computeSolution( problem = problem, uspline_bext = uspline_bext )
           gold_sol_coeff = numpy.array( [0.0, 2.45863125e-05, 4.92339375e-05, 4.92952500e-05] )
           self.assertTrue( numpy.allclose( test_sol_coeff, gold_sol_coeff ) )
           splineBarGalerkin.plotSolution( test_sol_coeff, uspline_bext )
           splineBarGalerkin.plotCompareFunToExactSolution( problem, test_sol_coeff, uspline_bext )