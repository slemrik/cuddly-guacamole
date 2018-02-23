import numpy as np
import system
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import rdf

def create_plot(sysbox, plot_optimisation, plot_simulation, plot_rdf):

    """ Creates various plots showing the evolution of the system configuration during the optimisation and simulation. 
    Args:
        sysbox (Box): the system object
        plot_rdf (Bool) whether or not to plot the rdf
    """

    fig = plt.figure()

    if plot_simulation:
        pos_history = sysbox.pos_history
        pot_history = sysbox.pot_history
    elif plot_optimisation:
        pos_history = sysbox.optimisation_pos_history
        pot_history = sysbox.optimisation_pot_history

    # 3D plot of the movement of the 0th particle in the system during simulation/optimisation:
    xyz = np.asarray(pos_history)[:,0] 
    ax_xyz = fig.add_subplot(321, projection='3d')
    ax_xyz.plot(*xyz.T, "--", color = "grey") 
    ax_xyz.scatter(*xyz.T, c = range(len(xyz))) # color shows whether point is from early or late in simulation

    # 2D plot of the xy-plane-projected movement of 0th particle during simulation/optimisation:
    xy = xyz[:,0:2] 
    ax_xy = fig.add_subplot(322)
    ax_xy.plot(*xy.T, "--", color = "grey") 
    ax_xy.scatter(*xy.T, c = range(len(xyz)))

    # 3D plot of the final system configuration (positions) of all particles in the system:
    xyz_final = np.asarray(pos_history[-1])
    ax_xyz_final = fig.add_subplot(323, projection='3d')
    ax_xyz_final.scatter(*xyz_final.T)

    # Plot of the evolution of the potential during the simulation/optimisation:
    ax_pot = fig.add_subplot(324)
    ax_pot.plot(np.asarray(pot_history).T)

    # Compute and plot radial density function:
    if plot_rdf:
        bsize = min(sysbox.size)
        r_Max = bsize / 2.5
        dr = r_Max/(35)
        rdff, radii, _ = rdf.pairCorrelationFunction_3D(pos_history[0][:,0], pos_history[0][:,1], pos_history[0][:,2], bsize, r_Max, dr) # rdf for 0th time step (to be averaged)
        for i in range(1, len(pos_history)):
            rdf2_tmp, _, _ = rdf.pairCorrelationFunction_3D(pos_history[i][:,0], pos_history[i][:,1], pos_history[i][:,2], bsize, r_Max, dr)
            rdff += rdf2_tmp
        rdff = rdff/len(pos_history) # average rdf over all timesteps (and particles<- taken care of in pairCorrelationFunction_3D)
        ax_rdf = fig.add_subplot(325)
        ax_rdf.plot(radii, rdff)    

    plt.show()
