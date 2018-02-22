import numpy as np
import system
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import rdf as rdff


def create_plot(bsize, pos_history, pot_history, r_cut_LJ, n_steps_opt, n_skip):
	
	no_of_bins = 30
	r_Max = r_cut_LJ*1.5
	shell_rad = np.linspace(0, r_Max, no_of_bins)
	radii = shell_rad[1:] 

	r_Max = r_cut_LJ*1/2
	dr = r_cut_LJ/30
	rdf, radii, _ = rdff.pairCorrelationFunction_3D(pos_history[0][:,0], pos_history[0][:,1], pos_history[0][:,2], bsize, r_Max, dr)
	for i in range(1, len(pos_history)):
	    rdf2_tmp, _, interior_indices = rdff.pairCorrelationFunction_3D(pos_history[i][:,0], pos_history[i][:,1], pos_history[i][:,2], bsize, r_Max, dr)
	    rdf += rdf2_tmp

	rdf = rdf/len(pos_history)
	print(rdf)
	print(radii)

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
	ax_xyz_final = fig.add_subplot(322, projection='3d') # plot of system configuration

	ax_xyz.plot(*xyz.T, "--", color = "grey") 
	ax_xyz.scatter(*xyz.T, c = range(len(xyz))) # color shows whether point is from early or late in simulation

	ax_xyz_final.scatter(*xyz_final.T)

	ax_pot = fig.add_subplot(323)
	ax_pot_late = fig.add_subplot(324)

	ax_pot.plot(np.asarray(pot_history).T)
	ax_pot_late.plot(np.asarray(pot_history[int(n_steps_opt/(1.2*n_skip)):]).T)

	ax_rdf = fig.add_subplot(325)
	ax_rdf.plot(radii, rdf)

	plt.show()
