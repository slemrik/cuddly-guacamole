import numpy as np
import neighbourlist
import system
import math
import numba as nb
from numba import cuda
from numba import vectorize 

p = 1e-6 #accuracy set to 1e-6 gives p??????
epsilon = 8.854187817e-12
#https://en.wikipedia.org/wiki/Vacuum_permittivity


@nb.jit(nopython = True)
def Ewald_short_energy_ij(r_ij,qi,qj,r_c):
    alpha = 1/(2**(1/2))/sigma(r_c)
    Ewald_energy_ij = qi*qj*1/(8*pi*epsilon) * math.erfc(alpha*r_ij)/r_ij
    
    return Ewald_energy_ij

#for long energy
@nb.jit(nopython = True)
def k_cut_off (r_cut,p) :
    k_c = (2*p)/r_cut 
    return k_c 

@nb.jit(nopython = True)
def sigma (r_c,p) : 
    sigma = r_c / np.sqrt (2*p) 
    return sigma 

@nb.jit(nopython = True)
def k_vectors(k_c, box): 
   k_vector = [ ]
   for k_i in range (-k_c,k_c+1):
       for k_j in range (-k_c,k_c+1):
           for k_k in range (-k_c,k_c+1):
                if np.linalg.norm(k_i,k_j,k_k) <= k_c : 
                #Reciprocal  lattice vector 
                    k = 2.0*np.pi*np.array([k_i / (box[0]), k_j / (box[1]), k_k / (box[2])] )   
                k_vector.append(k) 
   return k_vector 



@nb.jit(nopython = True)
def Ewald_long_energy(positions,EWald_neighbourlists,q,r_c,r_s,box):

    r_cut = r_c + r_s
    
    #prefactor 
    V=np.prod(box)
    pre_fac = 1/(2*V*epsilon)  

    k_c = k_cut_off (r_cut,p)
    sigma = sigma (r_cut,p)
    k_vector = k_vectors(k_c, box)

    

    if EWald_neighbourlists is None:
        # raise Exception('compute EWald_neighbourlists for particles before computing EWald energy!')
        return None

    


    ##
    for i in range(len(positions)):
        k = 0
        j = EWald_neighbourlists[i][k]  #NB LJneigbourlists[i] contains only the neighbours of particle i with indices j>i. 
                                    # Thus interactions are NOT counted twice by computing in this manner.
        while j!=-1: # -1 means no more neighbours in list
            r = np.linalg.norm(pbc.enforce_pbc_distance(positions[j] - positions[i], boxsize))
    
    Ewald_long_energy = 0.0
    str_factor = 0.0

    for i in range (len(k_vector)):  
        k = k_vector[i]
        #length of k
        k_length = np.linalg.norm (k) 
        k_length2 = k_length**2 
        
        for j in range (len(r)):
             charge = q[i]
             r_str = r [i] 
             str_fac += charge*np.cos(np.dot(k,r_str))
        

        exp_term += np.exp(-(sigma**2)*k_length2/2)/k_length2 
        Ewald_long_energy += np.linalg.norm(str_fac)**2*exp_term 
    
    Ewald_long_energy *= pre_fac 


    return Ewald_long_energy

@nb.jit(nopython = True)
def Ewald_self_energy_ij(r_ij,qi,qj,r_c):
    
    Ewald_energy_ij = 1/(2*epsilon*sigma(r_c)*(2*pi)**(3/2))(qi**2)
    return Ewald_energy_ij





@nb.jit(nopython = True)
def Ewald_enery(positions, EWald_neighbourlists, r_c, r_s):    
    '''Computes the total Lennard Jones potential of the system configuration of *box*.
    
    arguments:
        positions (numpy array): list of 3d numpy arrays of positions for each particle. 
        EWald_neighbourlists (numpy array): list of numpy arrays of integers of various sizes. EWald_neighbourlists[i] gives
        the indices of all particles that are in the neighbourlist of particle i in our system
        r_c (float): cutoff radius for Ewald_enery
        r_s (float): size of skin region for Ewald_energy
    '''


    if EWald_neighbourlists is None:
        # raise Exception('compute EWald_neighbourlists for particles before computing EWald energy!')
        return None

    r_cut = r_c + r_s

    Ewald_short_energy = 0.0
    
    Ewald_self_energy = 0.0

    for i in range(len(positions)):
        k = 0
        j = EWald_neighbourlists[i][k]  #NB LJneigbourlists[i] contains only the neighbours of particle i with indices j>i. 
                                    # Thus interactions are NOT counted twice by computing in this manner.
        while j!=-1: # -1 means no more neighbours in list
            r = np.linalg.norm(pbc.enforce_pbc_distance(positions[j] - positions[i], boxsize))
            
            Ewald_short_energy += Ewald_short_energy_ij(r,q[i],q[j],r_cut)
            Ewald_self_energy += Ewald_self_energy_ij(r,q[i],q[j],r_cut)
            
            k += 1
            j = EWald_neighbourlists[i][k]

    Ewald_long_energy = Ewald_long_energy(r,EWald_neighbourlists,q,r_c, r_s,box)
    return Ewald_short_energy + (1/np.dot(box)/epsilon)*Ewald_long_energy - Ewald_self_energy
