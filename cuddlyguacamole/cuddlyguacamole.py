import load
import system
import numpy as np
import pbc

temp = 120 #these could be somwhere more general place
dim = 3

r_cut_LJ = 6
r_skin_LJ = 0

bsize=22.56 #??????

n_opt_max = 50
n_steps_opt = 1000
tol_opt = 1/100
n_reuse_nblist = 1
n_skip = int(n_steps_opt/50)
width = bsize / 100
save_system_history = True

n_steps_sim = 5*n_steps_opt

def optimize(filename='sodium-chloride-example.npz'):
    print('optimize')
    parameters, positions, types, boxsize = load.load_input_file(filename)
    # print("parameters")
    # print(parameters)
    # print("positions")
    # mini = 100
    # for i in range(len(positions)-1):
    #     for j in range(i+1, len(positions)):
    #         mini = min(mini, np.linalg.norm(pbc.enforce_pbc_distance(positions[i]-positions[j], boxsize)))
    # print(mini) 
    # print(positions)
    # print("types")
    # print(types)
    # print("boxsize")
    # print(boxsize)
    particles = create_particle_object(parameters, positions, types)
    # boxsize = 5*np.ones(3) #remove it when size bug is solved
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