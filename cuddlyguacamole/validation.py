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


start = time.time()

############################################################################################################
# System and simulation setup:
############################################################################################################

kb = 1.38064852*10**(-23) # N*m/K (Boltzmann constant)
# xenon & argon: http://pbx-brasil.com/outrasDisciplinas/DinMol/Notas/IIarea/aula203/papers/PhysRev.159.98.pdf
sigma_argon = 1 # Å?
epsilon_argon = 1 # J(=Nm)? N*Å?
# sigma_xenon = 4.07 # Å
# epsilon_xenon = 225.3 # actually epsilon/kb (K) <- thus potential is actually potential/kb

dim = 3 # spatial dimension of system
bsize = 5
boxsize = np.ones(dim)*bsize # size of our system box
temperature = 120

r_cut_LJ = 2.5 # cut-off radius for LJ potential computation
n_steps = 1000 # no. of steps to simulate
n_reuse_nblist = 1 # update the neighbourlist for each particle only every n_reuse_nblist steps
n_skip = int(n_steps/50) # only save the system history every n_skip steps
width = bsize / 5 #r_cut_LJ / (n_reuse_nblist*20)
r_skin_LJ = 0 # skin region for LJ potential computation
n_particles = 64


############################################################################################################
# Generate input system configuration:
############################################################################################################


input.inputgenerator.gen_random_input_3D(filename = "input/testinput.csv", n_particles = n_particles, 
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

ourbox = system.Box(dimension = dim, size = boxsize, particles = particles, temp = temperature, kb = kb)
ourbox.compute_LJneighbourlist(r_cut_LJ, r_skin_LJ)
ourbox.compute_LJ_potential(r_cut_LJ, r_skin_LJ)

############################################################################################################
# Simulate system configuration evolution:
############################################################################################################

save_system_history = True
n_opt = 50
# tol_opt = ourbox.size[0]/100
# tol_opt = 1/50
# ourbox.optimize(n_opt, tol_opt, 20*int(n_steps/n_opt), n_reuse_nblist, n_skip, width, save_system_history, r_cut_LJ, r_skin_LJ)
ourbox.optimize(n_opt, n_steps, n_reuse_nblist, n_skip, width, save_system_history, r_cut_LJ, r_skin_LJ)
ourbox.simulate(n_steps*10, n_reuse_nblist, n_skip, width, save_system_history, r_cut_LJ, r_skin_LJ)
pos_history = ourbox.pos_history
# pot_history = kb*10**10*np.asarray(ourbox.pot_history) #kb*10**10 to get potential in Joule
pot_history = np.asarray(ourbox.pot_history) #kb*10**10 to get potential in Joule
# pot_increases = 0
# pot_decreases = 0
# for i, pot in enumerate(pot_history[1:]):
#     if pot_history[i] > pot_history[i-1]:
#         pot_increases += 1
#     elif pot_history[i] < pot_history[i-1]:
#         pot_decreases += 1
#     print(i)
#     print(pos)
#     print(pot)

# print(p_acc_vec)
# print(np.mean(p_acc_vec))
# print(pot_increases)
# print(pot_decreases)

entire_pot_history = []
for pot_history_opt_round_i in ourbox.optimisation_pot_history:
    for pot_step_j_opt_round_i in pot_history_opt_round_i[0:-1]:
        #entire_pot_history.append(kb*10**10*pot_step_j_opt_round_i)
        entire_pot_history.append(pot_step_j_opt_round_i)

entire_pot_history = np.asarray(entire_pot_history)

############################################################################################################
# Comparison with ideal gas configuration:
############################################################################################################

# Ideal gas density
no_of_distance_bins = 100
dist = np.linspace(0, np.sqrt(sum(boxsize**2)), no_of_distance_bins)

rho_igas = (n_particles-1) / (bsize**dim)
igas_particles_within_dist = rho_igas * 4 * np.pi * (dist[1:]**3 - dist[0:-1]**3) / 3

sim_particles_within_dist = np.zeros((len(pos_history), no_of_distance_bins-1), dtype=int)
dist_from_particle0 = np.zeros((len(pos_history), n_particles-1))
for i in range(len(pos_history)):
    for j in range(1, n_particles):
        dist_from_particle0[i][j-1] = np.linalg.norm(pos_history[i][j] - pos_history[i][0])

for i in range(len(pos_history)):
    for j in range(n_particles-1):
        for k in range(1, no_of_distance_bins):
            if dist_from_particle0[i][j] >= dist[k-1] and dist_from_particle0[i][j] < dist[k]:
                sim_particles_within_dist[i][k-1] += 1


mean_sim_particles_within_dist = np.mean(sim_particles_within_dist, axis = 0)
rdf = mean_sim_particles_within_dist / igas_particles_within_dist
print(dist[1:])
print(rdf)



############################################################################################################
# Plotting:
############################################################################################################

xyz = np.zeros((len(pos_history),3))
xy = np.zeros((len(pos_history), 2))
xyz_final = np.asarray(pos_history[-1])
for i, pos_i in enumerate(pos_history):
     xyz[i] = np.asarray(pos_i[0])
     xy[i] = np.asarray(pos_i[0][0:2])
xyz = np.asarray(xyz)
xy = np.asarray(xy)

fig = plt.figure()

ax_xyz = fig.add_subplot(321, projection='3d') # movement of 0th particle
# ax_xy = fig.add_subplot(222) # movement of 0th particle projected to xy-plane 
ax_xyz_final = fig.add_subplot(322, projection='3d') # plot of system configuration

ax_xyz.plot(*xyz.T, "--", color = "grey") 
ax_xyz.scatter(*xyz.T, c = range(len(xyz))) # color shows whether point is from early or late in simulation
# ax_xy.plot(*xy.T, "--", color = "grey") 
# ax_xy.scatter(*xy.T, c = pot_history)
ax_xyz_final.scatter(*xyz_final.T)

ax_pot = fig.add_subplot(323)
ax_pot_late = fig.add_subplot(324)

ax_pot.plot(np.asarray(pot_history).T * 1.38064852 * 10**(-23) / kb) # multiply to get energy in N*m
ax_pot_late.plot(np.asarray(pot_history[int(n_steps/(1.2*n_skip)):]).T * 1.38064852 * 10**(-23) / kb)

# ax_pot_full = fig.add_subplot(235)
# ax_pot_full.plot(np.asarray(entire_pot_history).T)


ax_rdf = fig.add_subplot(325)
ax_rdf.plot(dist[1:], rdf)


print(time.time() - start)

plt.show()



