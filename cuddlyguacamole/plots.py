import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import rdf

def plot_system_configuration(sysbox, plot_rdf):
    """ Creates various plots showing the evolution of the system configuration during the optimisation and simulation. 
    Args:
        sysbox (Box): the system object
        plot_rdf (Bool) whether or not to plot the rdf
    """

    pos_history = sysbox.pos_history
    pot_history = sysbox.pot_history
    entire_pot_history = optimisation_pot_history + pot_history

    fig = plt.figure()

    # 3D plot of the movement of the 0th particle in the system during simulation:
    xyz = np.asarray(pos_history)[:,0] 
    ax_xyz = fig.add_subplot(321, projection='3d')
    ax_xyz.plot(*xyz.T, "--", color = "grey") 
    ax_xyz.scatter(*xyz.T, c = range(len(xyz))) # color shows whether point is from early or late in simulation

    # 2D plot of the xy-plane-projected movement of 0th particle during simulation:
    xy = xyz[:,0:2] 
    ax_xy = fig.add_subplot(322)
    ax_xy.plot(*xy.T, "--", color = "grey") 
    ax_xy.scatter(*xy.T, c = range(len(xyz)))

    # 3D plot of the final system configuration (positions) of all particles in the system:
    xyz_final = np.asarray(pos_history[-1])
    ax_xyz_final = fig.add_subplot(323, projection='3d')
    ax_xyz_final.scatter(*xyz_final.T)

    # Plot of the evolution of the potential during the simulation:
    ax_pot = fig.add_subplot(324)
    ax_pot.plot(np.asarray(pot_history).T)

    # Plot of the evolution of the potential during the optimisation + simulation:
    ax_pot_full = fig.add_subplot(325) # plot of potential history during optimisation + simulation
    ax_pot_full.plot(np.asarray(entire_pot_history).T)

    # Compute and plot radial density function:
    if plot_rdf:
        bsize = min(sysbox.size)
        r_Max = bsize / 2
        dr = r_Max/(len(box.particles)/2)
        rdf, radii, _ = rdf.pairCorrelationFunction_3D(pos_history[0][:,0], pos_history[0][:,1], pos_history[0][:,2], bsize, r_Max, dr) # rdf for 0th time step (to be averaged)
        for i in range(1, len(pos_history)):
            rdf2_tmp, _, _ = rdf.pairCorrelationFunction_3D(pos_history[i][:,0], pos_history[i][:,1], pos_history[i][:,2], bsize, r_Max, dr)
            rdf += rdf2_tmp
        rdf = rdf/len(pos_history) # average rdf over all timesteps (and particles<- taken care of in pairCorrelationFunction_3D)
        ax_rdf = fig.add_subplot(326)
        ax_rdf.plot(radii, rdf)     
    

    plt.show()