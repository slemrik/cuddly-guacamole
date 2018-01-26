import numpy as np
import numpy.testing as npt
import lennardjones
import system    
import test.generate_test_system

def test_LJ_potential2(dim = 3, boxsize = 5*np.ones(3), nparticles = 2, temp = 120, charge = 1, sigma = 1, epsilon = 1, r_cut = 2.5, r_skin = 0):
    
    # Test with 5 particles located at [0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [1, 0, 0]:
    testbox = test.generate_test_system.generate_test_system(dim, boxsize*np.ones(dim), 5, temp, charge, sigma, epsilon)
    testbox.compute_LJneighbourlist(r_cut, r_skin)
    testbox.compute_LJ_potential(r_cut, r_skin)

    r01 = np.linalg.norm(np.array([0, 0, 1]) - np.array([0, 0, 0]))
    r02 = np.linalg.norm(np.array([0, 0, 2]) - np.array([0, 0, 0]))
    r03 = np.linalg.norm(np.array([0, 0, 3]) - np.array([0, 0, 0]))
    r04 = np.linalg.norm(np.array([1, 0, 0]) - np.array([0, 0, 0]))
    r12 = np.linalg.norm(np.array([0, 0, 2]) - np.array([0, 0, 1]))
    r13 = np.linalg.norm(np.array([0, 0, 3]) - np.array([0, 0, 1]))
    r14 = np.linalg.norm(np.array([1, 0, 0]) - np.array([0, 0, 1]))
    r23 = np.linalg.norm(np.array([0, 0, 3]) - np.array([0, 0, 2]))
    r24 = np.linalg.norm(np.array([1, 0, 0]) - np.array([0, 0, 2]))
    r34 = np.linalg.norm(np.array([1, 0, 0]) - np.array([0, 0, 3]))

    LJpot_01 = 4*epsilon*((sigma/r01)**12 - (sigma/r01)**6) - 4*epsilon*((sigma/(r_cut+r_skin))**12 - (sigma/(r_cut+r_skin))**6)
    LJpot_02 = 4*epsilon*((sigma/r02)**12 - (sigma/r02)**6) - 4*epsilon*((sigma/(r_cut+r_skin))**12 - (sigma/(r_cut+r_skin))**6)
    LJpot_03 = 0#4*epsilon*((sigma/r03)**12 - (sigma/r03)**6) - 4*epsilon*((sigma/(r_cut+r_skin))**12 - (sigma/(r_cut+r_skin))**6)
    LJpot_04 = 4*epsilon*((sigma/r04)**12 - (sigma/r04)**6) - 4*epsilon*((sigma/(r_cut+r_skin))**12 - (sigma/(r_cut+r_skin))**6)
    LJpot_12 = 4*epsilon*((sigma/r12)**12 - (sigma/r12)**6) - 4*epsilon*((sigma/(r_cut+r_skin))**12 - (sigma/(r_cut+r_skin))**6)
    LJpot_13 = 4*epsilon*((sigma/r13)**12 - (sigma/r13)**6) - 4*epsilon*((sigma/(r_cut+r_skin))**12 - (sigma/(r_cut+r_skin))**6)
    LJpot_14 = 4*epsilon*((sigma/r14)**12 - (sigma/r14)**6) - 4*epsilon*((sigma/(r_cut+r_skin))**12 - (sigma/(r_cut+r_skin))**6)
    LJpot_23 = 4*epsilon*((sigma/r23)**12 - (sigma/r23)**6) - 4*epsilon*((sigma/(r_cut+r_skin))**12 - (sigma/(r_cut+r_skin))**6)
    LJpot_24 = 4*epsilon*((sigma/r24)**12 - (sigma/r24)**6) - 4*epsilon*((sigma/(r_cut+r_skin))**12 - (sigma/(r_cut+r_skin))**6)
    LJpot_34 = 0#4*epsilon*((sigma/r34)**12 - (sigma/r34)**6) - 4*epsilon*((sigma/(r_cut+r_skin))**12 - (sigma/(r_cut+r_skin))**6)

    LJpot = LJpot_01 + LJpot_02 + LJpot_03 + LJpot_04 + LJpot_12 + LJpot_13 + LJpot_14 + LJpot_23 + LJpot_24 + LJpot_34

    npt.assert_almost_equal(LJpot, testbox.LJpotential, decimal = 7)





def test_LJ_potential_ij2(dim = 3, boxsize = 5*np.ones(3), nparticles = 2, temp = 120, charge = 1, sigma = 1, epsilon = 1, r_cut = 2.5, r_skin = 0):
    
    # Test with 5 particles located at [0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [1, 0, 0]:
    testbox = test.generate_test_system.generate_test_system(dim, boxsize*np.ones(dim), 5, temp, charge, sigma, epsilon)

    r01 = np.linalg.norm(np.array([0, 0, 1]) - np.array([0, 0, 0]))
    r02 = np.linalg.norm(np.array([0, 0, 2]) - np.array([0, 0, 0]))
    r03 = np.linalg.norm(np.array([0, 0, 3]) - np.array([0, 0, 0]))
    r04 = np.linalg.norm(np.array([1, 0, 0]) - np.array([0, 0, 0]))
    r12 = np.linalg.norm(np.array([0, 0, 2]) - np.array([0, 0, 1]))
    r13 = np.linalg.norm(np.array([0, 0, 3]) - np.array([0, 0, 1]))
    r14 = np.linalg.norm(np.array([1, 0, 0]) - np.array([0, 0, 1]))
    r23 = np.linalg.norm(np.array([0, 0, 3]) - np.array([0, 0, 2]))
    r24 = np.linalg.norm(np.array([1, 0, 0]) - np.array([0, 0, 2]))
    r34 = np.linalg.norm(np.array([1, 0, 0]) - np.array([0, 0, 3]))

    LJpot_01 = 4*epsilon*((sigma/r01)**12 - (sigma/r01)**6) - 4*epsilon*((sigma/(r_cut+r_skin))**12 - (sigma/(r_cut+r_skin))**6)
    LJpot_02 = 4*epsilon*((sigma/r02)**12 - (sigma/r02)**6) - 4*epsilon*((sigma/(r_cut+r_skin))**12 - (sigma/(r_cut+r_skin))**6)
    LJpot_03 = 4*epsilon*((sigma/r03)**12 - (sigma/r03)**6) - 4*epsilon*((sigma/(r_cut+r_skin))**12 - (sigma/(r_cut+r_skin))**6)
    LJpot_04 = 4*epsilon*((sigma/r04)**12 - (sigma/r04)**6) - 4*epsilon*((sigma/(r_cut+r_skin))**12 - (sigma/(r_cut+r_skin))**6)
    LJpot_12 = 4*epsilon*((sigma/r12)**12 - (sigma/r12)**6) - 4*epsilon*((sigma/(r_cut+r_skin))**12 - (sigma/(r_cut+r_skin))**6)
    LJpot_13 = 4*epsilon*((sigma/r13)**12 - (sigma/r13)**6) - 4*epsilon*((sigma/(r_cut+r_skin))**12 - (sigma/(r_cut+r_skin))**6)
    LJpot_14 = 4*epsilon*((sigma/r14)**12 - (sigma/r14)**6) - 4*epsilon*((sigma/(r_cut+r_skin))**12 - (sigma/(r_cut+r_skin))**6)
    LJpot_23 = 4*epsilon*((sigma/r23)**12 - (sigma/r23)**6) - 4*epsilon*((sigma/(r_cut+r_skin))**12 - (sigma/(r_cut+r_skin))**6)
    LJpot_24 = 4*epsilon*((sigma/r24)**12 - (sigma/r24)**6) - 4*epsilon*((sigma/(r_cut+r_skin))**12 - (sigma/(r_cut+r_skin))**6)
    LJpot_34 = 4*epsilon*((sigma/r34)**12 - (sigma/r34)**6) - 4*epsilon*((sigma/(r_cut+r_skin))**12 - (sigma/(r_cut+r_skin))**6)

    npt.assert_almost_equal(LJpot_01, lennardjones.LJ_potential_ij(r01, sigma, epsilon, sigma, epsilon, r_cut + r_skin), decimal = 6)
    npt.assert_almost_equal(LJpot_02, lennardjones.LJ_potential_ij(r02, sigma, epsilon, sigma, epsilon, r_cut + r_skin), decimal = 6)
    npt.assert_almost_equal(LJpot_03, lennardjones.LJ_potential_ij(r03, sigma, epsilon, sigma, epsilon, r_cut + r_skin), decimal = 6)
    npt.assert_almost_equal(LJpot_04, lennardjones.LJ_potential_ij(r04, sigma, epsilon, sigma, epsilon, r_cut + r_skin), decimal = 6)
    npt.assert_almost_equal(LJpot_12, lennardjones.LJ_potential_ij(r12, sigma, epsilon, sigma, epsilon, r_cut + r_skin), decimal = 6)
    npt.assert_almost_equal(LJpot_13, lennardjones.LJ_potential_ij(r13, sigma, epsilon, sigma, epsilon, r_cut + r_skin), decimal = 6)
    npt.assert_almost_equal(LJpot_14, lennardjones.LJ_potential_ij(r14, sigma, epsilon, sigma, epsilon, r_cut + r_skin), decimal = 6)
    npt.assert_almost_equal(LJpot_23, lennardjones.LJ_potential_ij(r23, sigma, epsilon, sigma, epsilon, r_cut + r_skin), decimal = 6)
    npt.assert_almost_equal(LJpot_24, lennardjones.LJ_potential_ij(r24, sigma, epsilon, sigma, epsilon, r_cut + r_skin), decimal = 6)
    npt.assert_almost_equal(LJpot_34, lennardjones.LJ_potential_ij(r34, sigma, epsilon, sigma, epsilon, r_cut + r_skin), decimal = 6)

    # npt.assert_almost_equal(nblist, test_nblist)

