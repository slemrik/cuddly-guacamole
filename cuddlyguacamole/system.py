import numpy as np
import lennardjones
import neighbourlist
import copy
import pbc
import metropolis

############################################################################
# Define Particle and Box classes 
############################################################################
class Particle(object):
    """An object of this class represents a single particle in our system.

    Attributes:
        position (numpy array of float): a 1-3 dimensional position vector
        charge (float): charge of the particle
        sigmaLJ (float): distance at which LJ potential is 0 for particle type
        epsilonLJ (float): depth of LJ potential well for particle type
    """

    def __init__(self, position, charge, sigmaLJ = 1.0, epsilonLJ = 1.0):
        self.position = position
        self.charge = charge
        self.sigmaLJ = sigmaLJ
        self.epsilonLJ = epsilonLJ


class Box(object):
    """An object of this class is a 1 to 3-dimensional square box in space, containing particles.

    Attributes:
        - dimension (int): spatial dimension of the system (1-3)
        - size (*dimension*-dimensional numpy array of float): 1d-3d numpy array giving size of the box in each direction
        - particles (list of Particle): list of particles in the box
        - positions (#particles x *dimension* numpy array of float): current position of all particles in the system
        - LJpotential (float): current total Lennard Jones potential of the system
        - temp (float): the system temperature
        - kb (float): value of the Boltzmann constant (which unit/value kb is given in decides the unit of the LJ potential). Default is kJ/(mol*K)
        - LJneighbourlists (NxN numpy array of int (N = # particles)): the leftmost part of row i lists the indices
            of the neighbouring particles of particle i (particles within the LJ cutoff radius). The rest of the row contains -1's.
        - sigmas (numpy array of float): array containing sigma for each particle in the system
        - epsilons (numpy array of float): array containing epsilon for each particle in the system 
        - r_c_LJ (float): cutoff radius for LJ potential calculation
        - r_skin_LJ (float): size of skin region for LJ potential calculation 
        - Cpotential (float): coulomb potential
        - pos_history (list of N x *dimension* numpy arrays (N = #particles)): element no. i (N*dim array) in the list holds the positions of 
            the particles at MC step i during the last simulation
        - pot_history (list of float): element no. i in the list holds the potential at MC step i during the last simulation
        - optimisation_pos_history: (list of N x *dimension* numpy arrays (N = #particles)): element no. i (N*dim array) in the list holds the positions of 
            the particles at MC step i during the optimisation round
        - optimisation_pot_history: (list of float):  element no. i in the list holds the potential at MC step i during the optimisation round
    """

    def __init__(self, dimension, size, particles, temp, kb = 0.008314462175):
        self.dimension = dimension
        self.size = size
        self.particles = particles
        self.LJpotential = None # to be computed when needed
        self.temp = temp
        self.kb = kb # the unit that kb is given in defines the units used by the system. F.ex. if units of kb is in N*Å/K, then energy must be in N*Å. 

        self.LJneighbourlists = None # to be computed when needed
        self.sigmas = np.zeros(len(particles))
        self.epsilons = np.zeros(len(particles))
        for i in range(len(self.sigmas)):
            self.sigmas[i] = particles[i].sigmaLJ
            self.epsilons[i] = particles[i].epsilonLJ

        self.Cpotential = None # to be computed when needed
        self.pos_history = None # to be filled in during simulation
        self.pot_history = None
        self.optimisation_pos_history = []
        self.optimisation_pot_history = []

        for particle in particles:
            particle.position = pbc.enforce_pbc_coordinates(particle.position, size) # make sure all given particle positions are inside the box - if not, fix them using PBC
        self.make_positions_list() # once all particles are given a position inside the box, initialise *positions*

    def compute_LJneighbourlist(self, r_cut, r_skin): # computes the (LJ) neighbourlists for the particles in the system
        self.LJneighbourlists = neighbourlist.verlet_neighbourlist(self.positions, r_cut, r_skin, self.size)

    def compute_LJ_potential(self, r_cut, r_skin): 
        self.LJpotential = lennardjones.LJ_potential(self.positions, self.LJneighbourlists, self.sigmas, self.epsilons, r_cut, r_skin, self.size)

    def make_positions_list(self): # update *positions* array based on position currently registered to each Particle object in *particles*
        self.positions = np.ones((len(self.particles), self.dimension))
        for i in range(len(self.particles)):
            self.positions[i] = self.particles[i].position 

    def update_particle_positions(self): # update position registered to each Particle object in *particles* based on *positions* array
        for i in range(len(self.particles)):
            self.particles[i].position = self.positions[i]

    def compute_energy(self, r_cutLJ, r_skinLJ, r_cutCo, r_skinCo): # computes both LJ and Coulomb potential of system
        self.compute_LJ_potential(r_cutLJ, r_skinLJ)
        self.compute_Coulomb_potential(r_cutCo, r_skinCo)

    def compute_Coulomb_potential(self, r_cutCo, r_rkin_Co):
        self.Cpotential = 0

    def simulate(self, n_steps, n_reuse_nblist, n_skip, width, save_system_history, r_cut_LJ, r_skin_LJ, r_cut_Co = 0, r_skin_Co = 0):
        """ Simulation method.

        Args:
            - self (Box): the system object itself
            - n_steps (int): how many MCMC steps to simulate
            - n_reuse_nblist (int): how many MCMC steps in a row to reuse the same neighbourlist
            - n_skip (int): how many steps to go between each time the system state is saved in *pot_history* and *pos_history*
            - width (float): approximat width (max change in single coordinate of the position) of the MCMC trial steps
            - save_system_history (bool): whether or not to store the system states during the configuration
            - r_cut_LJ (float): LJ cutoff radius
            - r_skin_LJ (float): size of LJ skin region
            - r_cut_Co (float): Coulomb cutoff radius(???)
            - r_skin_Co (float): Coulomb skin region (????)

        Method updates the system variables *positions*, *LJpotential*, *pos_history* and *pot_history*, *LJneighbourlists*, and also updates the position
        registered to each Particle object in *particles* by calling self.update_particle_positions().
        """

        self.positions, self.LJpotential, self.pos_history, self.pot_history = metropolis.mcmc(self, n_steps, width, n_skip, n_reuse_nblist, save_system_history, r_cut_LJ, r_skin_LJ, self.kb)
        self.update_particle_positions()
        self.compute_LJneighbourlist(r_cut_LJ, r_skin_LJ)

    def optimize(self, n_opt_max, n_steps, tol_opt, n_reuse_nblist, n_skip, width, save_system_history, r_cut_LJ, r_skin_LJ, r_cut_Co = 0, r_skin_Co = 0):
        """ Optimisation method.

        Args:
            - self (Box): the system object itself
            - n_opt_max (int): maximum amount of optimisation rounds to run (each run with decreasing temperature)
            - n_steps (int): how many MCMC steps to make for each optimisation round
            - tol_opt (float): optimisation will be terminated when the relative change in potential between optimisation round is smaller than tol_opt
            - n_reuse_nblist (int): how many MCMC steps in a row to reuse the same neighbourlist
            - n_skip (int): how many steps to go between each time the system state is saved in *pot_history* and *pos_history*
            - width (float): approximat width (max change in single coordinate of the position) of the MCMC trial steps
            - save_system_history (bool): whether or not to store the system states during the configuration
            - r_cut_LJ (float): LJ cutoff radius
            - r_skin_LJ (float): size of LJ skin region
            - r_cut_Co (float): Coulomb cutoff radius(???)
            - r_skin_Co (float): Coulomb skin region (????)

        Method updates the system variables *positions*, *LJpotential*, *optimisation_pos_history* and *optimisation_pot_history*, *LJneighbourlists*, 
        and also updates the position registered to each Particle object in *particles* by calling self.update_particle_positions().
        """

        original_temp = float(self.temp) # store original box temperature
        original_pos_history = list(self.pos_history) # store pos_history from last simulation  
        original_pot_history = list(self.pot_history)

        temp_decrease_factors = 1.0/(np.asarray(range(1, n_opt_max+1), dtype = float)) # reduce temperature from optimisation round i-1 to i by temp_decrease_factors[i]
        temperatures = self.temp * temp_decrease_factors
        
        i=0
        LJpotential_old = 1e16
        while np.abs((self.LJpotential - LJpotential_old)/self.LJpotential) > tol_opt and i < n_opt_max:
            LJpotential_old = self.LJpotential
            self.temp = temperatures[i] # decrease temperature before running simulation
            self.simulate(n_steps, n_reuse_nblist, n_skip, width, save_system_history, r_cut_LJ, r_skin_LJ)
            print("new potential = ", self.LJpotential)
            print("change = ", self.LJpotential - LJpotential_old)
            print("rel.change =", (self.LJpotential - LJpotential_old)/self.LJpotential, "\n -------")
            i += 1
            if save_system_history: # store system history
                self.optimisation_pos_history += list(self.pos_history) 
                self.optimisation_pot_history += list(self.pot_history)

            if np.abs((self.LJpotential - LJpotential_old)/self.LJpotential) == 0:
                raise ValueError('No change in potential during optimisation step. Optimisation stuck... Please try again.')    

        self.temp = original_temp # reset temperature  
        self.pos_history = original_pos_history # reset pos_history to the position history of the last simulation
        self.pot_history = original_pot_history      
        



