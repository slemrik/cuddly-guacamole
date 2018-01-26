import numpy as np
import numpy.testing as npt
import metropolis
import system    
import test.generate_test_system
import pbc




def test_mcmc2(dim = 3, boxsize = 5*np.ones(3), nparticles = 2, temp = 120, charge = 1, sigma = 1, epsilon = 1, r_cut = 2.5, r_skin = 0):
    
    # Test with 5 particles located at [0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [1, 0, 0]:
    testbox = test.generate_test_system.generate_test_system(dim, boxsize*np.ones(dim), 5, temp, charge, sigma, epsilon)
    testbox.compute_LJneighbourlist(r_cut, r_skin)
    testbox.compute_LJ_potential(r_cut, r_skin)








def test_mcmc_step2(dim = 3, boxsize = 5*np.ones(3), nparticles = 2, temp = 120, charge = 1, sigma = 1, epsilon = 1, r_cut = 2.5, r_skin = 0):
    
    # Test with 5 particles located at [0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [1, 0, 0]:
    testbox = test.generate_test_system.generate_test_system(dim, boxsize*np.ones(dim), 5, temp, charge, sigma, epsilon)
    testbox.compute_LJneighbourlist(r_cut, r_skin)
    testbox.compute_LJ_potential(r_cut, r_skin)


    pos_original = testbox.positions
    # print(testbox.positions)

    testbox.positions, accepted, trial_step, acceptance_prob = metropolis.mcmc_step(testbox, 1, r_cut, r_skin, True)

    # print(accepted)
    # print(acceptance_prob)
    # print(trial_step)
    # print(testbox.positions)

    positions_post_mcmc_step = np.array(pos_original)
    for i in range(len(positions_post_mcmc_step)):
        positions_post_mcmc_step[i] = pbc.enforce_pbc(pos_original[i] + trial_step[i]*float(accepted), boxsize*np.ones(dim))

    npt.assert_almost_equal(positions_post_mcmc_step, testbox.positions, decimal = 7)
