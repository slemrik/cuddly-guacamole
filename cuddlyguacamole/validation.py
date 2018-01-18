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
import sys

start = time.time()

############################################################################################################
# System and simulation setup:
############################################################################################################

# kb = 1.38064852*10**(-13) # N*Å/K (Boltzmann constant)
# xenon & argon: http://pbx-brasil.com/outrasDisciplinas/DinMol/Notas/IIarea/aula203/papers/PhysRev.159.98.pdf
# sigma_argon = 3.405 # Å
sigma_argon = 1
epsilon_argon = 1
# epsilon_argon = 119.8 # actually epsilon/kb (K) <- thus potential is actually potential/kb
# sigma_xenon = 4.07 # Å
# epsilon_xenon = 225.3 # actually epsilon/kb (K) <- thus potential is actually potential/kb

dim = 3 # spatial dimension of system
# boxsize = np.ones(dim)*sigma_argon*10 # size of our system box
boxsize = np.ones(dim)*20

# r_cut_LJ = 2.5*sigma_argon # cut-off radius for LJ potential computation
r_cut_LJ = 3
n_steps = 200 # no. of steps to simulate
n_reuse_nblist = 0 #int(n_steps/50) # update the neighbourlist for each particle only every n_reuse_nblist steps
n_skip = int(n_steps/100) # only save the system history every n_skip steps
width = r_cut_LJ / (n_reuse_nblist*20)
# r_skin_LJ = 2*n_reuse_nblist*width # skin region for LJ potential computation
r_skin_LJ = 0

############################################################################################################
# Generate input system configuration:
############################################################################################################


input.inputgenerator.gen_random_input_3D(filename = "input/testinput.csv", n_particles = 512, 
                                            boxsize = boxsize, r_c = r_cut_LJ + r_skin_LJ)

sysconfig = []

fid = open("input/testinput.csv","r")
fid_reader = csv.reader(fid, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
for row in fid_reader:
    sysconfig.append(row)

for i in range(len(sysconfig)):
    sysconfig[i][0:3] = pbc.enforce_pbc(sysconfig[i][0:3]*boxsize, boxsize)
    sysconfig[i][0:3] = pbc.enforce_pbc(np.random.rand(3), boxsize)


############################################################################################################
# Initialise particle list based on input and the system (ourbox):
############################################################################################################

particles = []

for i in range(len(sysconfig)):
    particles.append(system.Particle(position = np.asarray(sysconfig[i][0:3]), 
        charge = sysconfig[i][3], sigmaLJ = sigma_argon, epsilonLJ = epsilon_argon))

ourbox = system.Box(dimension = dim, size = boxsize, particles = particles, temp = 120.0)
ourbox.compute_LJneighbourlist(r_cut_LJ, r_skin_LJ)
ourbox.compute_LJ_potential(r_cut_LJ, r_skin_LJ)

############################################################################################################
# Simulate system configuration evolution:
############################################################################################################

save_system_history = True
sys.setrecursionlimit(5000) # to be able to run with many particles (deepcopy uses recursion https://docs.python.org/2/library/copy.html)
# ourbox, pos_history, pot_history, p_acc_vec = metropolis.mcmc(ourbox, n_steps, width, n_skip, n_reuse_nblist, save_system_history, r_cut_LJ, r_skin_LJ)
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

xyz = np.zeros((len(pos_history),3))
xy = np.zeros((len(pos_history), 2))
for i,pos_i in enumerate(pos_history):
     xyz[i] = np.asarray(pos_i[0])
     xy[i] = np.asarray(pos_i[0][0:2])
xyz = np.asarray(xyz)
xy = np.asarray(xy)

fig = plt.figure()

ax_xyz = fig.add_subplot(221, projection='3d')
ax_xy = fig.add_subplot(222)

ax_xyz.plot(*xyz.T, "--", color = "grey") 
ax_xyz.scatter(*xyz.T, c = pot_history)
ax_xy.plot(*xy.T, "--", color = "grey") 
ax_xy.scatter(*xy.T, c = pot_history)

ax_pot = fig.add_subplot(223)
ax_pot_late = fig.add_subplot(224)

ax_pot.plot(np.asarray(pot_history).T)
ax_pot_late.plot(np.asarray(pot_history[int(n_steps/(1.2*n_skip)):]).T)

print(pot_history)
print(time.time() - start)

plt.show()



