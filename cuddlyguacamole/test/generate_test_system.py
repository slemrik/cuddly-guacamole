import numpy as np
import numpy.testing as npt
import neighbourlist
import system    

def generate_test_system(dim = 3, boxsize = 5*np.ones(3), nparticles = 64, temp = 120, charge = 1, sigma = 1, epsilon = 1):

    if nparticles > 64:
        raise Exception('Max 64 particles in test system')

    # if min(boxsize) < 3:
    #     raise Exception('Boxsize minimum 3')

    g = [0,1,2,3]
    x,y,z = np.meshgrid(g,g,g)
    xyz = np.vstack((x.flat, y.flat, z.flat)).T
    xyz = np.ascontiguousarray(xyz)
    positions = xyz[0:nparticles]

    particles = []
    for i in range(nparticles):
        particles.append(system.Particle(position = positions[i], charge = charge, sigmaLJ = sigma, epsilonLJ = epsilon))

    testbox = system.Box(dimension = dim, size = boxsize, particles = particles, temp = temp)

    return testbox