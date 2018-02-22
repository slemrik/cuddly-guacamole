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

    def __init__(self, size, particles):
        self.r_cut = 2.5
        self.r_skin = 0
        self.size = size
        self.dimension = len(size)
        self.particles = particles
        self.temp = 120 #later resolve redundance with system object

        self.sigmas = np.zeros(len(particles))
        self.epsilons = np.zeros(len(particles))
        for i in range(len(self.sigmas)):
            self.sigmas[i] = particles[i].sigmaLJ
            self.epsilons[i] = particles[i].epsilonLJ

        self.positions = self.make_positions_list()
        self.LJneighbourlists = self.compute_LJneighbourlist()
        self.LJpotential = self.compute_LJ_potential() 
        self.Cpotential = self.compute_Coloumb_potential()

        for particle in particles:
            particle.position = pbc.enforce_pbc_coordinates(particle.position, size)
        self.make_positions_list()

    def make_positions_list(self): # update positions list based on position registered to each particle in particles
        positions = np.ones((len(self.particles), self.dimension))
        for i in range(len(self.particles)):
            positions[i] = self.particles[i].position # necessary to update box.positions and particle.position indidivually? or do they point to the same location in memory anyway by this assignment?
        return positions

    def update_particle_positions(self): # update registered position for each particle based on positions list (?)
        for i in range(len(self.particles)):
            self.particles[i].position = self.positions[i]

    def compute_energy(self, r_cutLJ, r_skinLJ, r_cutCo, r_skinCo):
        self.compute_LJ_potential(r_cutLJ, r_skinLJ)
        self.compute_Coloumb_potential(r_cutCo, r_skinCo)

    def compute_Coloumb_potential(self, r_cutCo=2.5, r_rkin_Co=0):
        return 0

    def compute_LJneighbourlist(self):
        return neighbourlist.verlet_neighbourlist(self.positions, self.r_cut, self.r_skin, self.size)

    def compute_LJ_potential(self):
        return lennardjones.LJ_potential(self.positions, self.LJneighbourlists, self.sigmas, self.epsilons, self.r_cut, self.r_skin, self.size)

        
class System(object):
    def __init__(self, box):

        self.box = box
        self.temp = 120
        self.n_reuse_nblist = 1
        self.save_system_history = True
        self.kb = 0.008314462175 # the unit that kb is given in defines the units used by the system. If units of kb is in N*Å/K, then energy is in N*Å.
        self.optimisation_pos_history = []
        self.optimisation_pot_history = []

        self.n_steps_opt = n_steps_opt = 1000 # no. of steps to simulate
        self.n_steps_sim = 5*n_steps_opt;
        self.n_opt_max = 30
        self.tol_opt = 1/100 #what is this
        self.n_skip = int(n_steps_opt/50) # only save the system history every n_skip steps
        bsize = 5 #it shoul be boxsize[0]
        self.width = bsize / 1000 #Why??
        self.n_reuse_nblist = 1 #not sure here, maybe it should be a given parameter

    def simulate(self):

        self.box.positions, self.box.LJpotential, self.pos_history, self.pot_history = metropolis.mcmc(self.box, self.n_steps_sim, self.width, self.n_skip, self.n_reuse_nblist, self.save_system_history, self.box.r_cut , self.box.r_skin , self.kb)
        self.box.update_particle_positions()
        self.box.LJneighbourlists = self.box.compute_LJneighbourlist()

    def optimize(self):
        original_temp = float(self.temp) # store original box temperature

        temp_decrease_factors = 1.0/(np.ones(self.n_opt_max)*range(1, self.n_opt_max+1)) # reduce temperature on each simulation i by temp_decrease_factors[i]
        temperatures = self.temp * np.ones(self.n_opt_max) * temp_decrease_factors
        
        i=0

        LJpotential_old = 1e16

        while np.abs((self.box.LJpotential - LJpotential_old)/self.box.LJpotential) > self.tol_opt and i < self.n_opt_max:
            # positions_tmp = self.positions
            LJpotential_old = self.box.LJpotential
            print('self.box.LJpotential', self.box.LJpotential)
            self.temp = temperatures[i]
            self.simulate()
            print("new potential = ", self.box.LJpotential)
            print("change = ", self.box.LJpotential - LJpotential_old)
            print("rel.change =", (self.box.LJpotential - LJpotential_old)/self.box.LJpotential, "\n -------")
            i += 1
            self.optimisation_pos_history.append(list(self.pos_history)) # store all position histories
            self.optimisation_pot_history.append(list(self.pot_history))

        self.temp = original_temp # reset temperature        

