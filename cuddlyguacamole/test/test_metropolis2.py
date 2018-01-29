import numpy as np
import numpy.testing as npt
import metropolis
import system    
import test.generate_test_system
import pbc




def test_mcmc2(dim = 3, boxsize = 5*np.ones(3), nparticles = 2, temp = 120, charge = 1, sigma = 1, epsilon = 1, r_cut = 2.5, r_skin = 0):
    kb = 1.38064852*10**(-23)
    # Test with 5 particles located at [0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [1, 0, 0]:
    testbox = test.generate_test_system.generate_test_system(dim, boxsize*np.ones(dim), 5, temp, charge, sigma, epsilon)
    testbox.compute_LJneighbourlist(r_cut, r_skin)
    testbox.compute_LJ_potential(r_cut, r_skin)

    mcmcbox = test.generate_test_system.generate_test_system(dim, boxsize*np.ones(dim), 5, temp, charge, sigma, epsilon)
    mcmcbox.compute_LJneighbourlist(r_cut, r_skin)
    mcmcbox.compute_LJ_potential(r_cut, r_skin)    

    n_steps = 100
    positions_vec_mcmc = []
    positions_vec_comp = []
    accepted_vec_mcmc = []
    accepted_vec_comp = []
    trial_step_vec_mcmc = []
    trial_step_vec_comp = []
    acceptance_prob_vec_mcmc = []
    acceptance_prob_vec_comp = []
    for i in range(n_steps):
        pos_mcmc_i, accepted_mcmc_i, trial_step_mcmc_i, acceptance_prob_mcmc_i = metropolis.mcmc_step(testbox, 1, r_cut, r_skin, True)
        if accepted_mcmc_i:
            testbox.positions = np.array(pos_mcmc_i)
            testbox.compute_LJneighbourlist(r_cut, r_skin)
            testbox.compute_LJ_potential(r_cut, r_skin)

        positions_vec_comp.append(pos_mcmc_i)
        accepted_vec_comp.append(accepted_mcmc_i)
        trial_step_vec_comp.append(trial_step_mcmc_i)
        acceptance_prob_vec_comp.append(acceptance_prob_mcmc_i)

    testingmatrix = list([np.asarray(accepted_vec_comp), np.asarray(trial_step_vec_comp)])
    mcmcbox.positions, mcmcbox.LJpotential, _, _, _ = metropolis.mcmc(mcmcbox, n_steps, 1, 1, 1, False, r_cut, r_skin, 1.38064852*10**(-23), True, testingmatrix)

    npt.assert_almost_equal(mcmcbox.positions, testbox.positions, decimal = 7)



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
