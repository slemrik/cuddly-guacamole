import load
import system
import numpy as np
import plot
import time

temp = 120 #these could be somwhere more general place
dim = 3

r_cut_LJ = 6
r_skin_LJ = 0

n_opt_max = 30
n_steps_opt = 1000
tol_opt = 1/100
n_reuse_nblist = 1
n_skip = int(n_steps_opt/50)
save_system_history = True

n_steps_sim = 5*n_steps_opt

def optimize(filename='sodium-chloride-example.npz'):
    start = time.time()
    print('optimize')
    parameters, positions, types, boxsize = load.load_input_file(filename)
    width = boxsize[1]/1000
    particles = create_particle_object(parameters, positions, types)
    box = system.Box(dim, boxsize, particles, temp)
    # return system.Box(dim, boxsize, particles, temp)

    box.compute_LJneighbourlist(r_cut_LJ, r_skin_LJ)
    box.compute_LJ_potential(r_cut_LJ, r_skin_LJ)
    box.optimize(n_opt_max, n_steps_opt, tol_opt, n_reuse_nblist, n_skip, width, save_system_history, r_cut_LJ, r_skin_LJ)
    box.simulate(n_steps_sim, n_reuse_nblist, n_skip, width, save_system_history, r_cut_LJ, r_skin_LJ)

    pos_history = box.pos_history
    pot_history = np.asarray(box.pot_history)

    plot.create_plot(boxsize[1], pos_history, pot_history, r_cut_LJ, n_steps_opt, n_skip)
    print('running time:',time.time() - start)


def simulate(filename='sodium-chloride-example.npz'):
    print('simulate')
    parameters, positions, types, boxsize = load.load_input_file(filename)
    width = boxsize[1]/1000
    particles = create_particle_object(parameters, positions, types)
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

optimize('sodium-chloride-example.npz')