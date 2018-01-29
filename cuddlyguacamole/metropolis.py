import numpy as np
import lennardjones
import neighbourlist
#import energy # for coloumb energy
import system
import numpy.testing as npt
import copy
import warnings
import pbc
import numba as nb

def mcmc_step(box, width, r_cut, r_skin, update_nblist, kb = 1.38064852*10**(-23)):

    positions_trial = np.zeros((len(box.particles), box.dimension))
    trial_step = width * np.random.randn(*positions_trial.shape)/4 #randn -> std norm. dist, divide by 4 to keep output mostly within (-0.5, 0.5)

    for i in range(len(positions_trial)):
        positions_trial[i] = pbc.enforce_pbc(box.positions[i] + trial_step[i], box.size) 
        
    particles_trial = [system.Particle(positions_trial[i], box.particles[i].charge, 
                                        box.particles[i].sigmaLJ, box.particles[i].epsilonLJ) for i in range(len(box.particles))] # set trial particle list with trial positions
    if update_nblist:
        nblist_trial = neighbourlist.verlet_neighbourlist(positions_trial, r_cut, r_skin) # update neighbourlist for trial positions 
    else:
        nblist_trial = np.array(box.LJneighbourlists)

    LJpotential_trial = lennardjones.LJ_potential(positions_trial, nblist_trial, box.sigmas, box.epsilons, r_cut, r_skin)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # acceptance_prob = min(1.0, np.exp(-(LJpotential_trial - box.LJpotential)/box.temp)) # note: potential is actually potential/kb
        acceptance_prob = min(1.0, np.exp(-(LJpotential_trial - box.LJpotential)/(kb*box.temp)))
        if LJpotential_trial - box.LJpotential > 0 and acceptance_prob > 0.001:
            print(LJpotential_trial - box.LJpotential)
            print(acceptance_prob)
        # print(acceptance_prob)

    # if (box_trial.LJpotential>box.LJpotential):
    #     print("increase = " + repr(box_trial.LJpotential-box.LJpotential))
    #     print("acceptance prob. = " + repr(acceptance_prob))

    if np.random.rand() < acceptance_prob:
        return positions_trial, True, trial_step, acceptance_prob # Return True for accepted trial step return trial_step and acceptance_prob to use in unit testing
    return box.positions, False, trial_step, acceptance_prob # return False for rejected trial step (no need to update box object)


def mcmc(box, n_steps, width, n_skip, n_reuse_nblist, save_system_history, r_cut_LJ, r_skin_LJ, kb = 1.38064852*10**(-23), testing = False, testingmatrix=None):
    '''Metropolis MCMC simulation of the movement of the particles within *box*.
    NB: Currently only using LJ potential. Returns the box object with new system configuration
    and a system configuration history.
    
    arguments:
        box (Box): our box object 
        n_steps (int): how many mcmc steps in total to make
        width (float): approximate "max" size of each mcmc step
        n_skip (int): we only save the system state for every n_skip steps
        n_reuse_nblist (int): how many mcmc steps to do before updating the LJ
                                verlet neighbourlist of the particles in *box*
        save_system_history (bool): save the history of system states during the mcmc simulation? 
        r_cut_LJ (float): cut-off radius for LJ neighbourlist computation
        r_skin_LJ (float): size of skin-reigon for LJ neighbourlist computation
    '''
    # Store initial position for each particle in list
    positions_history = [np.array(box.positions)]             
    # (Compute then) Store initial total LJ potential:
    if box.LJpotential is None:
        box.compute_LJ_potential(r_cut_LJ, r_skin_LJ)
    potLJ_history = [box.LJpotential]

    p_acc_vec = [] # for testing...
    for i in range(int(np.ceil(n_steps/n_skip))): #?
        for j in range(n_skip):
            update_nblist = (np.mod(i*n_skip+j, n_reuse_nblist+1) == 0)
            if testing:
                box.positions = np.array(box.positions + testingmatrix[1][i*n_skip+j]*float(testingmatrix[0][i*n_skip + j]))
            else:
                positions_new, accepted, _, p_acc = mcmc_step(box, width, r_cut_LJ, r_skin_LJ, update_nblist) # mcmc acceptance prob p_acc used in testing            
                if accepted:
                    box.positions = np.array(positions_new)
                    box.update_particle_positions()
                    if update_nblist:
                        box.compute_LJneighbourlist(r_cut_LJ, r_skin_LJ)
                    box.compute_LJ_potential(r_cut_LJ, r_skin_LJ)
        # if (box.LJpotential>box_old.LJpotential):
        #     print("increase = " + repr(box.LJpotential-box_old.LJpotential))
        #     print("acceptance prob. = " + repr(p_acc))
        if save_system_history:
            positions_history.append(np.array(box.positions))
            potLJ_history.append(box.LJpotential)
            p_acc_vec.append(p_acc)
        # print(i*n_skip+n_skip)

    return box.positions, box.LJpotential, positions_history, potLJ_history, p_acc_vec # return history and p_acc_vec for use in testing



