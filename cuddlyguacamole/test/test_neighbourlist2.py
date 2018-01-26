import numpy as np
import numpy.testing as npt
import neighbourlist
import system    
import matplotlib.pyplot as plt
import test.generate_test_system

def test_verlet_neighbourlist2(dim = 3, boxsize = 5*np.ones(3), nparticles = 5, temp = 120, charge = 1, sigma = 1, epsilon = 1, r_cut = 2.5, r_skin = 0):
    
    # Test with 5 particles located at [0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [1, 0, 0], [1, 0, 1], [1, 0, 2]:
    testbox = test.generate_test_system.generate_test_system(dim, boxsize*np.ones(dim), 7, temp, charge, sigma, epsilon) # nparticles = 7
    testbox.compute_LJneighbourlist(r_cut, r_skin)

    # Neighbourlist should then be: 
    nblist = np.array([[1, 2, 4, 5, 6, -1, -1], 
                       [2, 3, 4, 5, 6, -1, -1 ],
                       [3, 4, 5, 6, -1, -1, -1],
                       [5, 6, -1, -1, -1, -1, -1],
                       [5, 6, -1, -1, -1, -1 ,-1],
                       [6, -1, -1, -1, -1, -1, -1],
                       [-1, -1, -1, -1, -1, -1, -1]])

    npt.assert_equal(nblist, testbox.LJneighbourlists)

