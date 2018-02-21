import numpy as np
import neighbourlist
import system
import pbc
import numba as nb

@nb.jit(nopython = True)
def LJ_potential_ij(r, sigmaii, epsilonii, sigmajj, epsilonjj, r_cut):
    
    sigmaij = 0.5 * (sigmaii + sigmajj) # Lorentz-Berthelot: https://en.wikipedia.org/wiki/Combining_rules
    epsilonij = np.sqrt(epsilonii * epsilonjj)

    q = (sigmaij / r)**6
    q_c = (sigmaij/r_cut)**6
    LJ_pot_ij = (4.0 * epsilonij * q * (q - 1.0)      # return lennard jones interaction energy between particle i and particle j
               - 4.0 * epsilonij * q_c *(q_c - 1.0))  # subtract value of potential at r_cut to avoid discontinuity # precompute!!!!!!
    return LJ_pot_ij


@nb.jit(nopython = True)
def LJ_potential(positions, LJneighbourlists, sigmas, epsilons, r_c, r_s, boxsize):    
    '''Computes the total Lennard Jones potential of the system configuration of *box*.
    
    arguments:
        positions (numpy array): list of 3d numpy arrays of positions for each particle. 
        LJneighbourlists (numpy array): list of numpy arrays of integers of various sizes. LJneighbourlists[i] gives
        the indices of all particles that are in the neighbourlist of particle i in our system
        sigmas (numpy array of float): sigma for each particle
        epsilons (numpy array of float): epsilon for each particle  
        r_c (float): cutoff radius for LJ potential
        r_s (float): size of skin region for LJ potential
    '''


    if LJneighbourlists is None:
        # raise Exception('compute LJneighbourlists for particles before computing LJ potential!')
        return None

    r_cut = r_c + r_s

    LJpot = 0.0
    for i in range(len(positions)):
        k = 0
        j = LJneighbourlists[i][k]  #NB LJneigbourlists[i] contains only the neighbours of particle i with indices j>i. 
                                    # Thus interactions are NOT counted twice by computing in this manner.
        while j!=-1: # -1 means no more neighbours in list
            r = np.linalg.norm(pbc.enforce_pbc_distance(positions[j] - positions[i], boxsize))
            # print(i,j)
            # print(positions[j],positions[i])
            LJpot += LJ_potential_ij(r, sigmas[i], epsilons[i], sigmas[j], epsilons[j], r_cut)
            k += 1
            j = LJneighbourlists[i][k]

    return LJpot


