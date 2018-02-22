import numpy as np
np.set_printoptions(threshold=np.inf)
import numpy.testing as npt
import neighbourlist
import system    
import pbc
import random

def generate_test_system(dim = 3, boxsize = 5*np.ones(3), nparticles = 64, temp = 120, charge = 1, sigma = 1, epsilon = 1, kb = 0.008314462175):

    no_of_gridpoints = int(np.cbrt(nparticles))+1
    g = np.array(range(-no_of_gridpoints, no_of_gridpoints+1)) # generate a list of 3-dimensional pos vectors with combinations of -no_of_gridpoints-1, ...,-1, 0, 1, ..., no_of_gridpoints+1
    while g[-1] >= min(boxsize/2):
        g = g/2
    x,y,z = np.meshgrid(g,g,g)
    xyz = np.vstack((x.flat, y.flat, z.flat)).T
    xyz = np.ascontiguousarray(xyz)

    positions = sorted(xyz, key=lambda k: random.random())[0:nparticles]

    # for i in range(len(positions)):
    #     tmp_pos = np.array(positions[i])
    #     positions[i] = pbc.enforce_pbc_coordinates(positions[i], boxsize)    

    particles = []
    for i in range(nparticles):
        particles.append(system.Particle(position = positions[i], charge = charge, sigmaLJ = sigma, epsilonLJ = epsilon))

    testbox = system.Box(dimension = dim, size = boxsize, particles = particles, temp = temp, kb = kb)

    return testbox