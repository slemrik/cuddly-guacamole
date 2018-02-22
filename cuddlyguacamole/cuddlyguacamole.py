import load
import system
import numpy as np

temp = 120 #these could be somwhere more general place
dim = 3

r_cut_LJ = 2.5
r_skin_LJ = 0

bsize=5 #??????

n_opt_max = 30
n_steps_opt = 1000
tol_opt = 1/100
n_reuse_nblist = 1
n_skip = int(n_steps_opt/50)
width = bsize / 1000
save_system_history = True

n_steps_sim = 5*n_steps_opt

def optimize(filename='sodium-chloride-example.npz'):
    print('optimize')
    parameters, positions, types, boxsize = load.load_input_file(filename)
    particles = create_particle_object(parameters, positions, types)
    boxsize = 5*np.ones(3) #remove it when size bug is solved
    box = system.Box(dim, boxsize, particles, temp)

    box.compute_LJneighbourlist(r_cut_LJ, r_skin_LJ)
    box.compute_LJ_potential(r_cut_LJ, r_skin_LJ)
    box.optimize(n_opt_max, n_steps_opt, tol_opt, n_reuse_nblist, n_skip, width, save_system_history, r_cut_LJ, r_skin_LJ)

def simulate(filename='sodium-chloride-example.npz'):
    print('simulate')
    parameters, positions, types, boxsize = load.load_input_file(filename)
    particles = create_particle_object(parameters, positions, types)
    boxsize = 5*np.ones(3) #remove it when size bug is solved
    box = system.Box(dim, boxsize, particles, temp)

    box.compute_LJneighbourlist(r_cut_LJ, r_skin_LJ)
    box.compute_LJ_potential(r_cut_LJ, r_skin_LJ)
    box.simulate(n_steps_sim, n_reuse_nblist, n_skip, width, save_system_history, r_cut_LJ, r_skin_LJ)

def create_particle_object(parameters, positions, types):
    particles = []

    for i in range(len(types)):
        position = positions[i]
        sigmaLJ = parameters[types[i]][0]
        epsilonLJ = parameters[types[i]][1]
        charge = parameters[types[i]][3]

        particles.append(system.Particle(position, charge, sigmaLJ, epsilonLJ))

    return particles

# optimize('sodium-chloride-example.npz')