import numpy as np
import lennardjones
import neighbourlist
#import energy # for coloumb energy
import system
import warnings
import pbc
import numba as nb

def mcmc_step(box, width, r_cut, r_skin, update_nblist, kb = 0.008314462175):
    '''Executes single step of the MCMC simulation (see mcmc below).
    NB: Currently only using LJ potential. 
    
    Arguments:
        box (Box): the system/box object 
        width (float): approximate maximum size (change in individual coordinate) of each mcmc step
        r_cut (float): cut-off radius for LJ neighbourlist computation
        r_skin (float): size of skin-reigon for LJ neighbourlist computation
        kb (float): boltzmann constant. Default is given in kJ/(mol*K). Can be provided in different units if desired (will result in different units for the potential and distances...)
        
    Returns:
        box.positions (numpy array): the updated (if trial step was accepted) array of positions of the particles in the system
        _ (Bool): an unnamed boolean which relays whether or not the trial step was accepted
    '''    

    positions_trial = np.zeros((len(box.particles), box.dimension))
    trial_step = width * np.random.randn(*positions_trial.shape)/4 #randn -> std norm. dist, divide by 4 to keep output mostly within (-0.5, 0.5)

    for i in range(len(positions_trial)):
        positions_trial[i] = pbc.enforce_pbc_coordinates(box.positions[i] + trial_step[i], box.size)

    if update_nblist:
        nblist_trial = neighbourlist.verlet_neighbourlist(positions_trial, r_cut, r_skin, box.size) # update neighbourlist for trial positions 
    else:
        nblist_trial = np.array(box.LJneighbourlists)

    LJpotential_trial = lennardjones.LJ_potential(positions_trial, nblist_trial, box.sigmas, box.epsilons, r_cut, r_skin, box.size)

    with warnings.catch_warnings(): # suppress warnings for possibly large arguments of the exponential function
        warnings.simplefilter("ignore")
        acceptance_prob = min(1.0, np.exp(-(LJpotential_trial - box.LJpotential)/(kb*box.temp)))

    if np.random.rand() < acceptance_prob:
        return positions_trial, True # Return True for accepted trial step
    return box.positions, False 


def mcmc(box, n_steps, width, n_skip, n_reuse_nblist, save_system_history, r_cut_LJ, r_skin_LJ, kb = 0.008314462175):
    '''Metropolis MCMC simulation of the movement of the particles within *box*.
    
    Arguments:
        box (Box): the system/box object 
        n_steps (int): how many mcmc steps in total to make
        width (float): approximate maximum size (change in individual coordinate) of each mcmc step
        n_skip (int): determines how often the system state (state is saved in vector positions_history every n_skip steps)
        n_reuse_nblist (int): how many mcmc steps to make before updating the verlet neighbourlist of the particles in *box*
        save_system_history (bool): save the history of system states during the mcmc simulation? 
        r_cut_LJ (float): cut-off radius for LJ neighbourlist computation
        r_skin_LJ (float): size of skin-reigon for LJ neighbourlist computation
        kb (float): boltzmann constant. Default is given in kJ/(mol*K). Can be provided in different units if desired (will result in different units for the potential and distances...)
        
    Returns:
        box.positions (numpy array): the updated array of positions of the particles in the system
        box.LJpotential (float): the updated LJ potential resulting from the new system configuration
        positions_history (list of numpy array): a list holding the particle positions for each state the mc went through
        potLJ_history (list of float): a list holding the history of the LJ potential fo the mcmc
    '''
    # Store initial position for each particle in list
    positions_history = [np.array(box.positions)]             
    # (Compute then) Store initial total LJ potential:
    if box.LJpotential is None:
        box.compute_LJ_potential(r_cut_LJ, r_skin_LJ)
    potLJ_history = [box.LJpotential]

    for i in range(int(np.ceil(n_steps/n_skip))): #?
        for j in range(n_skip):
            update_nblist = (np.mod(i*n_skip+j, n_reuse_nblist+1) == 0)
            positions_new, accepted = mcmc_step(box, width, r_cut_LJ, r_skin_LJ, update_nblist, kb)    
            if accepted:
                box.positions = np.array(positions_new)
                box.update_particle_positions()
                if update_nblist:
                    box.compute_LJneighbourlist(r_cut_LJ, r_skin_LJ)
                box.compute_LJ_potential(r_cut_LJ, r_skin_LJ)
        if save_system_history:
            positions_history.append(np.array(box.positions))
            potLJ_history.append(box.LJpotential)

    return box.positions, box.LJpotential, positions_history, potLJ_history


