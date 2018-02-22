import load
import system
import numpy as np

def optimize(filename='sodium-chloride-example.npz'):
    print('optimize')
    parameters, positions, types, boxsize = load.load_input_file(filename)
    particles = create_particle_object(parameters, positions, types)
    boxsize = 5*np.ones(3) #remove it when size bug is solved
    box = system.Box(boxsize, particles)

    our_system = system.System(box)
    our_system.optimize()

def simulate(filename='sodium-chloride-example.npz'):
    print('simulate')
    parameters, positions, types, boxsize = load.load_input_file(filename)
    particles = create_particle_object(parameters, positions, types)
    boxsize = 5*np.ones(3) #remove it when size bug is solved
    box = system.Box(boxsize, particles)

    our_system = system.System(box)
    our_system.simulate()

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