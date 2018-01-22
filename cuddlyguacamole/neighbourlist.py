import numpy as np
import system
import numba as nb

@nb.jit#(nopython = True)
def verlet_neighbourlist(positions, r_cut, r_skin):
    """Verlet neighbourlist computation: for each of the particles in the array *box.particles*,
    we compute a neighbourlist of particles that lie within a cutoff radius r_cut+r_skin. 
    Returns an updated list of numpy arrays neighbourlists where each array contains the neighbourlist for that particle.

    arguments:
        particles (list of Particle): listing of Particle objects in the system
        r_cut: verlet cutoff radius
        r_skin: size of verlet skin region
    """

    N = len(positions)
    r_cutt = r_cut + r_skin
    neighbourlists = np.full((N, N), -1)  # initialise entries of nblist array i to 0. 
                                                                                # The indices of the actual neighbours of particle i
                                                                                # will be in the beginning of the array, before the -1's

    for i in range(N):
        k = 0 # counts no. of entries in  nblist of particle i
        for j in range(len(positions[i+1:])): # NB: by only considering positions[i+1:] only particle j will 
                                                        # appear in the nblist of particle i but particle i will not 
                                                        # appear in the nblist of j... thus the neighbourlist is not always a 
                                                        # true neighbourlist for each individual particle. But it does record of 
                                                        # each pair of neighbours in any case 
            if np.linalg.norm(positions[i] - positions[j + i+1]) < r_cutt:
                neighbourlists[i][k] =  int(j + i+1)
                k += 1

    return neighbourlists

