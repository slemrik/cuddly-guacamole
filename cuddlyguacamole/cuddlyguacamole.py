import load
import system

temp = 120 #these could be somwhere more general place
dim = 3

def optimize(filename='sodium-chloride-example.npz'):
    print('optimizer')
    parameters, positions, types, boxsize = load.load_input_file(filename)
    particles = create_particle_object(parameters, positions, types)
    box = system.Box(dim, boxsize, particles, temp)
    return box

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