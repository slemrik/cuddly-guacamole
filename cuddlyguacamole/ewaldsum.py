import numpy as np
import neighbourlist
import system
import math
import pbc
import numba as nb
from numba import cuda
from numba import vectorize 

p = 10 #accuracy set to 1e-6 gives p??????
epsilon = 8.854187817e-12
#https://en.wikipedia.org/wiki/Vacuum_permittivity

#Computes the total Ewald_short_energy for single step.
@nb.jit(nopython = True)
def Ewald_short_energy_ij(r_ij,qi,qj,r_cut):
    
    '''
    arguments:
        r_ij (numpy array): the distance between ri & rj
        qi: the chage of qj (+1 or -1)
        qj: the charge of qj (+1 or -1) 
        r_c (float): cutoff radius for Ewald
    '''

    alpha = 1/(2**(1/2))/cal_sigma(r_cut) #the factor in the Ewald
    Ewald_energy_ij = qi*qj*1/(8*np.pi*epsilon) * math.erfc(alpha*r_ij)/r_ij
    
    return Ewald_energy_ij

#Computes the total Ewald short energy
@nb.jit(nopython = True)
def Ewald_short_energy(positions, Ewald_neighbourlists, r_cut):    
    '''
    arguments:
        positions (numpy array): list of 3d numpy arrays of positions for each particle. 
        EWald_neighbourlists (numpy array): list of numpy arrays of integers of various sizes. EWald_neighbourlists[i] gives
        the indices of all particles that are in the neighbourlist of particle i in our system
        r_c (float): cutoff radius for Ewald_enery
        r_s (float): size of skin region for Ewald_energy
    '''

    if Ewald_neighbourlists is None:
        # raise Exception('compute EWald_neighbourlists for particles before computing EWald energy!')
        return None

   

    Ewald_short_energy = 0.0

    for i in range(len(positions)):
        k = 0
        j = Ewald_neighbourlists[i][k]  #NB EWald_neighbourlists[i] contains only the neighbours of particle i with indices j>i. 
                                    # Thus interactions are NOT counted twice by computing in this manner.
        while j!=-1: # -1 means no more neighbours in list
            r = np.linalg.norm(pbc.enforce_pbc_distance(positions[j] - positions[i], boxsize))
            
            Ewald_short_energy += Ewald_short_energy_ij(r,q[i],q[j],r_cut)
            
            k += 1
            j = Ewald_neighbourlists[i][k]

    return Ewald_short_energy


#for long energy
@nb.jit(nopython = True)
def k_cut_off (r_cut) :
    '''Calculate r_cut to k_cut

    arguments:
        r_c (float): cutoff radius for Ewald
    '''
    k_cut = (2*p)/r_cut
    return k_cut 

@nb.jit(nopython = True)
def cal_sigma (r_cut) : 
    '''Calculte sigma parameter with r_cut and p
    
    arguments:
        r_c (float): cutoff radius for Ewald
    '''
    sigma = r_cut / np.sqrt (2*p) 
    return sigma 

@nb.jit(nopython = True)
def cal_k_vectors(k_c, box): 
   '''Generate k_vectors with k_cutoff and box
    
    arguments:
        k_c (float): k space cutoff radius for Ewald
        box: box vector
    '''
   k_vector = [ ]
   for k_i in range (-k_c,k_c+1):
       for k_j in range (-k_c,k_c+1):
           for k_k in range (-k_c,k_c+1):
                if np.linalg.norm(k_i,k_j,k_k) <= k_c : 
                #Reciprocal  lattice vector 
                    k = 2.0*np.pi*np.array([k_i / (box[0]), k_j / (box[1]), k_k / (box[2])] )   
                k_vector.append(k) 
   return k_vector 


#Calculte total Ewald long range energy
@nb.jit(nopython = True)
def Ewald_long_energy(positions,EWald_neighbourlists,q,r_cut,box):
    '''
    arguments:
        positions: the chage value
        r_c (float): cutoff radius for Ewald
    '''
    
    
    
    #prefactor & parameters
    V=np.prod(box)
    pre_fac = 1/(2*V*epsilon)  

    k_c = k_cut_off (r_cut)
    sigma = cal_sigma (r_cut)
    k_vector = cal_k_vectors(k_c, box)


     #create the counted r array
    if Ewald_neighbourlists is None:
        # raise Exception('compute EWald_neighbourlists for particles before computing EWald energy!')
        return None
    r_number = 0 #the number of counted position (r)
    for i in range(len(positions)):
        k = 0
        j = Ewald_neighbourlists[i][k]  #NB EWald_neighbourlists[i] contains only the neighbours of particle i with indices j>i. 
                                    # Thus interactions are NOT counted twice by computing in this manner.
        while j!=-1: # -1 means no more neighbours in list
            r_number += 1

            k += 1
            j = Ewald_neighbourlists[i][k]
    

    #calculte the long energy  
    Ewald_long_energy = 0.0
    str_factor = 0.0

    for i in range (len(k_vector)):  
        k = k_vector[i]
        #length of k
        k_length = np.linalg.norm (k) 
        k_length2 = k_length**2 
        
        for j in range (len(r_number)):
             charge = q[i]
             r_str = r [i] 
             str_fac += charge*np.cos(np.dot(k,r_str))
        

        exp_term += np.exp(-(sigma**2)*k_length2/2)/k_length2 
        Ewald_long_energy += (1/np.dot(box)/epsilon)*np.linalg.norm(str_fac)**2*exp_term 
    
    Ewald_long_energy = pre_fac * str_fac * exp_term 


    return Ewald_long_energy

#Calculte single step of Ewald self energy
@nb.jit(nopython = True)
def Ewald_self_energy_ij(q,r_cut):    
    '''
    arguments:
        q: the charge value
        r_c (float): cutoff radius for Ewald
    '''
    Ewald_energy_ij = 1/(2 * epsilon * cal_sigma(r_c) * (2* math.pi)**(3/2))*(q**2)
    return Ewald_energy_ij

#Calculte total Ewald self energy
@nb.jit(nopython = True)
def Ewald_self_energy(positions,q,r_cut):
    '''
    arguments:
        q: the chage value
        r_c (float): cutoff radius for Ewald
    '''
    
    Ewald_self = 0
    for i in range(len(positions)):
        Ewald_self += Ewald_self_energy_ij(q[i],r_cut)
    

    return Ewald_self

@nb.jit(nopython = True)
def Ewald_energy(positions,EWald_neighbourlists,q,r_c,r_s,boxsize):
    r_cut = r_c + r_s    
    
    Ewald_short = Ewald_short_energy(positions, Ewald_neighbourlists, r_cut)
    Ewald_long = Ewald_long_energy(positions,EWald_neighbourlists,q,r_cut,box)
    Ewald_self = Ewald_self_energy(positions,q,r_cut)
    
    return Ewald_short + Ewald_long - Ewald_self
