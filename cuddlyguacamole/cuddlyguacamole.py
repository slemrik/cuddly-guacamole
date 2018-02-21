import load
import system

def optimize(filename):
    print('optimizer')
    parameters, positions, types = load.load_input_file(filename)
    create_particle_object(parameters, positions, types)

def create_particle_object(parameters, positions, types):
    particles = []

    for i in range(len(types)):
        position = positions[i]
        sigmaLJ = parameters[types[i]][0]
        epsilonLJ = parameters[types[i]][1]
        charge = parameters[types[i]][3]

        particles.append(system.Particle(position, charge, sigmaLJ, epsilonLJ))

optimize('sodium-chloride-example.npz')