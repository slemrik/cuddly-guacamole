import numpy as np
import system
import numba as nb
import pbc

@nb.jit#(nopython = True)
def verlet_neighbourlist(positions, r_cut, r_skin, boxsize):
    """Neighbourlist computation.

    Arguments:
        positions (N x dimension numpy array of float (N=#particles)): lists positions of particles in system
        r_cut (float): verlet cutoff radius
        r_skin (float): size of verlet skin region
        boxsize (float): size of system box (for computing distances using pbc)

    Returns:
        neighbourlists (NxN array of int): the leftmost part of neigbourlists[i] (row i in the array) lists the indices
            of the "neighbouring particles" (particles within the cutoff radius) of particle i [NB! neigbourlists[i] contains only the neighbours
            of particle i with indices j>i! Thus neighbourlists does contain all pairs of neighbours, but a particular row may not contain all the neighbours
            of that particle]. The rest of row i contains -1's, thus once a -1 is encountered in the row, one knows there are no more neighbours. 
    """
    N = len(positions)
    r_cutt = r_cut + r_skin
    neighbourlists = np.full((N, N), -1)  # initialise entries of nblist array i to 0. 
                                                                                # The indices of the actual neighbours of particle i
                                                                                # will be in the beginning of the array, before the -1's
    for i in range(N):
        k = 0 # counts no. of entries in  nblist of particle i
        for j in range(i+1, N): # NB: by only considering positions[i+1:] only particle j will 
                                                        # appear in the nblist of particle i but particle i will not 
                                                        # appear in the nblist of j... thus the neighbourlist is not always a 
                                                        # true neighbourlist for each individual particle. But it does record of 
                                                        # each pair of neighbours in any case 
            if np.linalg.norm(pbc.enforce_pbc_distance(positions[i] - positions[j], boxsize)) < r_cutt:
                neighbourlists[i][k] =  j
                k += 1

    return neighbourlists

