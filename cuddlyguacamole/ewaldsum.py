#Mathematical derivation. with reference to  http://micro.stanford.edu/mediawiki/images/4/46/Ewald_notes.pdf 

import numpy as np
import neighbourlist
import system
from numpy.linalg import norm
from scipy.special import erfc
from scipy import exp, pi

"""
    i : potential location
    r : list of radii
    q : list of charges
    alpha : Ewald parameter
    cutoff_rspace : real space box cutoff
    cutoff_kspace : Fourier space box cutoff
 """

def energy(i, r, q, cell, cutoff_rspace, cutoff_kspace):

    Energy_short  = short_energy(i, r, q, cell, cutoff_rspace)
    Energy_long   = long_energy(i, r, q, cutoff_kspace,)
    Energy_self   = self_energy(i, r, q, cell)
    
    return Energy_short+Energy_long-Engery_self



def potential(i, r, q, cell, area, invcell, alpha, cutoff_rspace, cutoff_kspace):

    Vr = short_energy_sum(i, r, q, cell, alpha, cuttoff_rspace)
    Vf = total_long_energy(i, ,r, q, invcell, alpha, cutoff_kpsace, area)
    Vs = total_s_energy(i, r, q, cell, alpha)
    
    return Vr+Vf+Vs


#energy calculation formula. with reference to equation 39 in page 7 of the pdf(link found in first line of this file). 
def short_energy_sum (i, r, q, cell, alpha, cutoff_rspace):
    
    for j in range(0, len(q)):
        
   
        
        
    return Vr


def total_long_energy (i, r, q, cutoff_kspace):
    long_e = 0
    
    for j in range(0, len(q)):
            
    long_pre = 1/(2*V*epsilon_0)     #prefactor of the long-ranged term         
    k                                #????reciprocal vector need to be defined 
    k2 = k**2 
    sigma2 = sigma** 2 
    s_k = sum( q*np.exp(k*r))         #structure factor 
    s_k2= s_k**2                      #sqaure of structure factor    

    midpart = np.sum(exp(-sigma2*k2/2)/k2) 
     
        total_long_energy = long_pre*midpart*s_k2
    return total_long_energy


def total_s_energy(self):                                 #general algo. done 
    s_energy = 0 
    s_total_potential = 0 
    s_pre = 1/(4*pi*eplison_0*sigma*np.sqrt(2*np.pi))     #prefactor of the self-term
              
    for i in range(0, len(q)):                            # ==1  particle code to be corrected to system.  
        total_s_potential += s_potential(i)               # !! def self_potential 
      
    total_s_energy = s_pre*s_total_potential
         
    return total_s_energy

def str_fac (self) :
    s_k=0 
    for i in range ():
        s_k += charge(i)*exp(k*position(i))
    return s_k 
