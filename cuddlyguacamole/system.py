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
        dimension (int): spatial dimension of the system (1-3)
        size (*dimension*-dimensional numpy array of float): 1d-3d numpy array giving size of the box in each direction
        particles (list of Particle): list of particles in the box
        positions (#particles x *dimension* numpy array of float): current position of all particles in the system
        LJpotential (float): current total Lennard Jones potential of the system
        temp (float): temperature in the box
        kb (float): value of the Boltzmann constant (which unit/value kb is given in decides the unit of the LJ potential). Default is kJ/(mol*K)
        LJneighbourlists (numpy array of numpy arrays of int (i.e. NxN of int where N = # particles)): 
                                                     an array of same size as *particles*,
                                                     with a numpy array of int for each particle listing the indices of the 
                                                     neighbouring particles (particles within the LJ cutoff radius)
        epsilons (numpy array of float): array containing epsilon for each particle in box
        sigmas (numpy array of float): array containing sigma for each particle in box
        r_c_LJ (float): cutoff radius for LJ potential calculation
        r_skin_LJ (float): size of skin region for LJ potential calculation 
    """

    def __init__(self, dimension, size, particles, temp, optimisation_pos_history=[], optimisation_pot_history = [], kb = 1.38064852*10**(-23)):
        self.dimension = dimension
        self.size = size
        self.particles = particles
        self.LJpotential = None
        self.temp = temp
        self.kb = kb # the unit that kb is given in defines the units used by the system. If units of kb is in N*Å/K, then energy is in N*Å. 

        self.LJneighbourlists = None
        self.sigmas = np.zeros(len(particles))
        self.epsilons = np.zeros(len(particles))
        for i in range(len(self.sigmas)):
            self.sigmas[i] = particles[i].sigmaLJ
            self.epsilons[i] = particles[i].epsilonLJ

        self.Cpotential = None
        self.pos_history = None
        self.pot_history = None
        self.optimisation_pos_history = optimisation_pos_history # variable to keep all position histories throughout the optimisation
        self.optimisation_pot_history = optimisation_pot_history

        for particle in particles:
            particle.position = pbc.enforce_pbc_coordinates(particle.position, size)
        self.make_positions_list()

    def compute_LJneighbourlist(self, r_cut, r_skin):
        self.LJneighbourlists = neighbourlist.verlet_neighbourlist(self.positions, r_cut, r_skin, self.size)

    def compute_LJ_potential(self, r_cut, r_skin):
        self.LJpotential = lennardjones.LJ_potential(self.positions, self.LJneighbourlists, self.sigmas, self.epsilons, r_cut, r_skin, self.size)

    def make_positions_list(self): # update positions list based on position registered to each particle in particles
        self.positions = np.ones((len(self.particles), self.dimension))
        for i in range(len(self.particles)):
            self.positions[i] = self.particles[i].position # necessary to update box.positions and particle.position indidivually? or do they point to the same location in memory anyway by this assignment?

    def update_particle_positions(self): # update registered position for each particle based on positions list (?)
        for i in range(len(self.particles)):
            self.particles[i].position = self.positions[i]

    def compute_energy(self, r_cutLJ, r_skinLJ, r_cutCo, r_skinCo):
        self.compute_LJ_potential(r_cutLJ, r_skinLJ)
        self.compute_Coloumb_potential(r_cutCo, r_skinCo)

    def compute_Coloumb_potential(self, r_cutCo, r_rkin_Co):
        self.Cpotential = 0

    def simulate(self, n_steps, n_reuse_nblist, n_skip, width, save_system_history, r_cut_LJ, r_skin_LJ, r_cut_Co = 0, r_skin_Co = 0):
        self.positions, self.LJpotential, self.pos_history, self.pot_history = metropolis.mcmc(self, n_steps, width, n_skip, n_reuse_nblist, save_system_history, r_cut_LJ, r_skin_LJ, self.kb)
        self.update_particle_positions()
        self.compute_LJneighbourlist(r_cut_LJ, r_skin_LJ)

    def optimize(self, n_opt_max, n_steps, tol_opt, n_reuse_nblist, n_skip, width, save_system_history, r_cut_LJ, r_skin_LJ, r_cut_Co = 0, r_skin_Co = 0):
        original_temp = float(self.temp) # store original box temperature

        temp_decrease_factors = 1.0/(np.ones(n_opt_max)*range(1, n_opt_max+1)) # reduce temperature on each simulation i by temp_decrease_factors[i]
        temperatures = self.temp * np.ones(n_opt_max) * temp_decrease_factors
        
        i=0
        # positions_tmp = [np.ones(self.dimension) for x in range(len(self.positions))]
        # positions_tmp[0] = 1e15*np.ones(self.dimension) # give positions_tmp[0] some large arbitrary value to pass the first while check
        # while np.linalg.norm(np.asarray(self.positions) - np.asarray(positions_tmp)) > tol_opt and i < n_opt:
        LJpotential_old = 1e16
        while np.abs((self.LJpotential - LJpotential_old)/self.LJpotential) > tol_opt and i < n_opt_max:
            # positions_tmp = self.positions
            LJpotential_old = self.LJpotential
            self.temp = temperatures[i]
            self.simulate(n_steps, n_reuse_nblist, n_skip, width, save_system_history, r_cut_LJ, r_skin_LJ)
            print("new potential = ", self.LJpotential)
            print("change = ", self.LJpotential - LJpotential_old)
            print("rel.change =", (self.LJpotential - LJpotential_old)/self.LJpotential, "\n -------")
            i += 1
            self.optimisation_pos_history.append(list(self.pos_history)) # store all position histories
            self.optimisation_pot_history.append(list(self.pot_history))

        self.temp = original_temp # reset temperature        
        



