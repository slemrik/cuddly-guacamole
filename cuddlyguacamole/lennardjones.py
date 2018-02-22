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
               - 4.0 * epsilonij * q_c *(q_c - 1.0))  # subtract value of potential at r_cut to avoid discontinuity (precompute?)
    return LJ_pot_ij


@nb.jit(nopython = True)
def LJ_potential(positions, LJneighbourlists, sigmas, epsilons, r_c, r_s, boxsize):    
    '''Computes the total Lennard Jones potential for the system configuration given by *positions*.
    
    Arguments:
        positions (Nx3 numpy array (N=#particles) of float): array holding position for each particle in system 
        LJneighbourlists (NxN numpy array of int (N = # particles)): the leftmost part of LJneigbourlists[i] (row i in the array) lists the indices
            of the "neighbouring particles" (particles within the LJ cutoff radius) of particle i [NB! LJneigbourlists[i] contains only the neighbours
            of particle i with indices j>i! Thus LJneighbourlists does contain all pairs of neighbours, but a particular row may not contain all the neighbours
            of that particle]. The rest of the row contains -1's. 
        sigmas (numpy array of float): sigma for each particle
        epsilons (numpy array of float): epsilon for each particle  
        r_c (float): cutoff radius for LJ potential
        r_s (float): size of skin region for LJ potential
    '''
    if LJneighbourlists is None:
        # raise Exception('Compute LJneighbourlists for particles before computing LJ potential!')
        return None

    r_cut = r_c + r_s

    LJpot = 0.0
    for i in range(len(positions)):
        k = 0
        j = LJneighbourlists[i][k]  # NB! LJneigbourlists[i] contains only the neighbours of particle i with indices j>i. 
                                    # Thus interactions are NOT counted twice by computing in the manner below.
        while j!=-1: # -1 means no more neighbours in list
            r = np.linalg.norm(pbc.enforce_pbc_distance(positions[j] - positions[i], boxsize))
            LJpot += LJ_potential_ij(r, sigmas[i], epsilons[i], sigmas[j], epsilons[j], r_cut)
            k += 1
            j = LJneighbourlists[i][k]

    return LJpot

#     import numpy as np
# import neighbourlist
# import system
# import pbc
# import numba as nb

# @nb.jit(nopython = True)
# def LJ_potential_ij(r, sigmaii, epsilonii, sigmajj, epsilonjj):#, r_cut):
#     '''Computes and returns the Lennard Jones potential between two particles at distance r.
#     '''

#     sigmaij = 0.5 * (sigmaii + sigmajj) # Lorentz-Berthelot: https://en.wikipedia.org/wiki/Combining_rules
#     epsilonij = np.sqrt(epsilonii * epsilonjj)

#     q = (sigmaij / r)**6
#     # q_c = (sigmaij/r_cut)**6
#     LJ_pot_ij = (4.0 * epsilonij * q * (q - 1.0))      # return lennard jones interaction energy between particle i and particle j
#                # - 4.0 * epsilonij * q_c *(q_c - 1.0))  # subtract value of potential at r_cut to avoid discontinuity (precompute??)
#     return LJ_pot_ij


# @nb.jit(nopython = True)
# def LJ_potential(positions, LJneighbourlists, sigmas, epsilons, boxsize):#, r_c, r_s):    
#     '''Computes the total Lennard Jones potential for the system configuration given by *positions*.
    
#     Arguments:
#         positions (Nx3 numpy array (N=#particles) of float): array holding position for each particle in system 
#         LJneighbourlists (NxN numpy array of int (N = # particles)): the leftmost part of LJneigbourlists[i] (row i in the array) lists the indices
#             of the "neighbouring particles" (particles within the LJ cutoff radius) of particle i [NB! LJneigbourlists[i] contains only the neighbours
#             of particle i with indices j>i! Thus LJneighbourlists does contain all pairs of neighbours, but a particular row may not contain all the neighbours
#             of that particle]. The rest of the row contains -1's. 
#         sigmas (numpy array of float): sigma for each particle
#         epsilons (numpy array of float): epsilon for each particle  
#         r_c (float): cutoff radius for LJ potential
#         r_s (float): size of skin region for LJ potential

#     Returns:
#         LJpot (float): the total LJ potential
#     '''
#     if LJneighbourlists is None:
#         # raise Exception('Compute LJneighbourlists for particles before computing LJ potential!')
#         return None

#     # r_cut = r_c + r_s
#     LJpot = 0.0
#     for i in range(len(positions)):
#         k = 0
#         j = LJneighbourlists[i][k]  # NB! LJneigbourlists[i] contains only the neighbours of particle i with indices j>i. 
#                                     # Thus interactions are NOT counted twice by computing in the manner below.
#         while j!=-1: # -1 means no more neighbours in list
#             r = np.linalg.norm(pbc.enforce_pbc_distance(positions[j] - positions[i], boxsize))
#             LJpot += LJ_potential_ij(r, sigmas[i], epsilons[i], sigmas[j], epsilons[j])#, r_cut)
#             k += 1
#             j = LJneighbourlists[i][k]

#     return LJpot


