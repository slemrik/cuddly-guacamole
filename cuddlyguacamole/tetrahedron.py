import numpy as np
import csv
import system
import input.inputgenerator
import metropolis
import neighbourlist
import lennardjones
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import pbc


############################################################################################################
# System and simulation setup:
############################################################################################################

# kb = 1.38064852*10**(-13) # N*Å/K (Boltzmann constant)
# xenon & argon: http://pbx-brasil.com/outrasDisciplinas/DinMol/Notas/IIarea/aula203/papers/PhysRev.159.98.pdf
sigma_argon = 3.405 # Å
epsilon_argon = 119.8 # actually epsilon/kb (K) <- thus potential is actually potential/kb
# sigma_xenon = 4.07 # Å
# epsilon_xenon = 225.3 # actually epsilon/kb (K) <- thus potential is actually potential/kb

dim = 3 # spatial dimension of system
boxsize = np.ones(dim)*sigma_argon*10 # size of our system box

r_cut_LJ = 2.5*sigma_argon # cut-off radius for LJ potential computation
n_steps = 200 # no. of steps to simulate
n_reuse_nblist = int(n_steps/50) # update the neighbourlist for each particle only every n_reuse_nblist steps
n_skip = int(n_steps/200) # only save the system history every n_skip steps
width = r_cut_LJ / (n_reuse_nblist*20)
r_skin_LJ = 2*n_reuse_nblist*width # skin region for LJ potential computation


############################################################################################################
# Generate input system configuration:
############################################################################################################


input.inputgenerator.gen_random_input_3D(filename = "input/testinput.csv", n_particles = 4, 
                                            boxsize = boxsize, r_c = r_cut_LJ + r_skin_LJ)

sysconfig = []

fid = open("input/testinput.csv","r")
fid_reader = csv.reader(fid, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
for row in fid_reader:
    sysconfig.append(row)

for i in range(len(sysconfig)):
    sysconfig[i][0:3] = pbc.enforce_pbc(sysconfig[i][0:3]*boxsize, boxsize)


############################################################################################################
# Initialise particle list based on input and the system (ourbox):
############################################################################################################

particles = []

for i in range(len(sysconfig)):
    particles.append(system.Particle(position = np.asarray(sysconfig[i][0:3]), 
        charge = sysconfig[i][3], sigmaLJ = sigma_argon, epsilonLJ = epsilon_argon))

ourbox = system.Box(dimension = dim, size = boxsize, particles = particles, temp = 20)
ourbox.compute_LJneighbourlist(r_cut_LJ, r_skin_LJ)
ourbox.compute_LJ_potential(r_cut_LJ, r_skin_LJ)

############################################################################################################
# Simulate system configuration evolution:
############################################################################################################

save_system_history = True
ourbox.simulate(n_steps, n_reuse_nblist, n_skip, width, save_system_history, r_cut_LJ, r_skin_LJ)

# print(ourbox.LJpotential)
pos_history = ourbox.pos_history
pot_history = ourbox.pot_history
pot_increases = 0
pot_decreases = 0
for i, pot in enumerate(pot_history[1:]):
    if pot_history[i] > pot_history[i-1]:
        pot_increases += 1
    elif pot_history[i] < pot_history[i-1]:
        pot_decreases += 1
#     print(i)
#     print(pos)
#     print(pot)

# print(p_acc_vec)
# print(np.mean(p_acc_vec))
# print(pot_increases)
# print(pot_decreases)

############################################################################################################
# Plotting:
############################################################################################################

xyz = pos_history[-1]
# xyz.append(xyz[0])
# xyz.append(xyz[2])
# xyz.append(xyz[1])
# xyz.append(xyz[3])
xyz = np.asarray(xyz)

fig = plt.figure()

ax_xyz = fig.add_subplot(111, projection='3d')

ax_xyz.plot(*xyz.T, "--", color = "grey") 
ax_xyz.scatter(*xyz.T)

edges = np.zeros((3,4,3))
for i in range(3):
    for j in range(i+1,4):
        edges[i][j][:] = xyz[j] - xyz[i]

angles = []
for i in range(3):
    for j in range(i+1,3):
        for k in range(j+1,4): 
            angles.append(np.arccos(np.inner(edges[i][j][:], edges[i][k][:]) / (np.linalg.norm(edges[i][j][:]) * np.linalg.norm(edges[i][k][:]))) * 180 / np.pi)

print(xyz)
print(edges)
print(angles)

plt.show()



