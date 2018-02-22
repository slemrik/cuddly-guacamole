import load
import system
import numpy as np
import plot
import time


r_cut_LJ = 2.5*(3.3284+4.40104)/2
n_opt_max = 30
n_steps_opt = 1000
tol_opt = 1/500
n_reuse_nblist = 50
n_skip = int(n_steps_opt/50)
n_steps_sim = 5*n_steps_opt

def optimize(filename='sodium-chloride-example.npz', dim=3, temp=120, width = 0, tol_opt = 1/500, ):

    print('optimize')
    parameters, positions, types, boxsize = load.load_input_file(filename)
    width = boxsize[1]/1000
    particles = create_particle_object(parameters, positions, types)
    print(parameters)
    box = system.Box(dim, boxsize, particles, temp)
    # return system.Box(dim, boxsize, particles, temp)


    if width == 0: # if no width specified
        width = boxsize[0]/5000
    r_skin_LJ = 2*n_reuse_nblist*np.linalg.norm(width*np.ones(3))
    save_system_history = True


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

    pos_history = box.pos_history
    pot_history = np.asarray(box.pot_history)

    plot.create_plot(boxsize[1], pos_history, pot_history, r_cut_LJ, n_steps_opt, n_skip)
    print('running time:',time.time() - start)

def create_particle_object(parameters, positions, types):
    particles = []

    for i in range(len(types)):
        position = positions[i]
        sigmaLJ = parameters[types[i]][0]
        epsilonLJ = parameters[types[i]][1]
        charge = parameters[types[i]][3]

        particles.append(system.Particle(position, charge, sigmaLJ, epsilonLJ))

    return particles
