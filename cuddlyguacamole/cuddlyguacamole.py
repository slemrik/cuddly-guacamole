import load
import system
import numpy as np
import plot
import time

def optimize(box = None, filename=None, 
                dim=3, temp=200, width = None, tol_opt = 1/250, 
                n_opt_max = 50, n_steps_opt = 500, n_reuse_nblist = None,
                n_skip = None, save_system_history = True, to_plot = True, plot_rdf = False):
    '''Optimises configuration of input system using Metropolis MCMC.
    
    Arguments:
        - box (Box): optional input system object of Box class (see system.py)
        - filename (str): path to input .npz file with input data (either box or filename must be provided) (see sodium-chloride-example.npz for example of input data)
        - dim (int): spatial dimension of the system
        - temp (float): system temperature in K 
        - width (float): approximate max. change in single coordinate of a particle position in one MCMC step
        - tol_opt (float): relative error tolerance in the potential. Optimisation terminates once tol_opt is reached 
        - n_opt_max (int): maximum number of optimisation rounds to run
        - n_steps_opt (int): MCMC steps per optimisation round
        - n_reuse_nblist (int): how many MCMC steps in a row to use the same neighbourlist for (determines the size of the skin region through r_skin = 2*n_reuse_nblist*d where d is approx. max distance of single MCMC step)
        - n_skip (int): only store the system state ever n_skip steps 
        - save_system_history (bool): whether or not to store the system states during optimisation
        - to_plot (bool): whether or not to plot evolution of system configuration for the optimisation 
        - plot_rdf (bool): whether or not to also plot the radial density functions

    Returns:
        box (Box): updated/newly created system object of Box class (includes copy of the
            system configuration history for the optimisation as box.optimisation_pos_history and box.optimisation_pot_history if save_system_history is True)
    '''
    start = time.time()
    ###########################################################################
    #  Set default values for parameters which depend on other input parameters. 
    ###########################################################################

    if n_reuse_nblist is None:
        n_reuse_nblist = max(1, int(n_steps_opt / 200))

    if n_skip is None:
        n_skip = max(1, int(n_steps_opt/200))

    ###########################################################################
    #  Load input data and create system.
    ###########################################################################
    
    # Check that input data is given correctly:
    if box is None and filename is None:
        raise InputError('Please provide input data.')
    if box is not None and filename is not None:
        raise InputError('Please provide either input data file OR input data Box object, not both.')

    # Load input and create system:
    box, r_cut_LJ, r_skin_LJ, width = load_system(filename, box, dim, temp, width, n_reuse_nblist)

    ###########################################################################
    # Run optimisation:
    ###########################################################################
    print('optimize')
    box.optimize(n_opt_max, n_steps_opt, tol_opt, n_reuse_nblist, n_skip, width, save_system_history, r_cut_LJ, r_skin_LJ)
    
    if to_plot:
        plot.create_plot(box, True, False, plot_rdf)
    
    print('running time:',time.time() - start)
    return box


def simulate(box = None, filename=None, 
                dim=3, temp=200, width = None, n_reuse_nblist = None, 
                n_steps_sim = 2000, n_skip = None, save_system_history = True, to_plot = True, plot_rdf = False):
    start = time.time()
    '''Simulates evolution of system configuration in time using Metropolis MCMC.
    
    Arguments:
        - box (Box): optional input system object of Box class (see system.py)
        - filename (str): path to input .npz file with input data (either box or filename must be provided) (see sodium-chloride-example.npz for example of input data)
        - dim (int): spatial dimension of the system
        - temp (float): system temperature in K 
        - width (float): approximate max. change in single coordinate of a particle position in one MCMC step
        - n_reuse_nblist (int): how many MCMC steps in a row to use the same neighbourlist for (determines the size of the skin region through r_skin = 2*n_reuse_nblist*d where d is approx. max distance of single MCMC step)        
        - n_steps_sim (int): how many MCMC steps to simulate
        - n_skip (int): only store the system state ever n_skip steps 
        - save_system_history (bool): whether or not to store the system states during optimisation
        - to_plot (bool): whether or not to plot evolution of system configuration for the optimisation 
        - plot_rdf (bool): whether or not to also plot the radial density functions

    Returns:
        box (Box): updated/newly created system object of Box class (includes copy of the
            system configuration history for the simulation as box.pos_history and box.pot_history if save_system_history is True)
    '''
    ###########################################################################
    #  Set default values for parameters which depend on other input parameters. 
    ###########################################################################

    if n_reuse_nblist is None:
        n_reuse_nblist = max(1, int(n_steps_sim / 200))

    if n_skip is None:
        n_skip = max(1, int(n_steps_sim/200))

    ###########################################################################
    #  Load input data and create system.
    ###########################################################################
    
    # Check that input data is given correctly:
    if box is None and filename is None:
        raise InputError('Please provide input data.')
    if box is not None and filename is not None:
        raise InputError('Please provide either input data file OR input data Box object, not both.')

    # Load input and create system:
    box, r_cut_LJ, r_skin_LJ, width = load_system(filename, box, dim, temp, width, n_reuse_nblist)

    ###########################################################################
    # Run simulation:
    ###########################################################################
    print('simulate')
    box.simulate(n_steps_sim, n_reuse_nblist, n_skip, width, save_system_history, r_cut_LJ, r_skin_LJ)

    if to_plot:
        plot.create_plot(box, False, True, plot_rdf)
    
    print('running time:',time.time() - start)
    return box


def load_system(filename, box, dim, temp, width, n_reuse_nblist):
    
    if box is None: # Load system data from file:
        parameters, positions, types, boxsize = load.load_input_file(filename)
        particles = create_particle_object(parameters, positions, types)
        box = system.Box(dim, boxsize, particles, temp)

        # Set size of r_cut_LJ = 2.5*sigma
        r_cut_LJ = 2.5*np.mean(box.sigmas)

        if width is None: # if no width (~MCMC step size) specified
            width = boxsize[0]/5000

        # Set size of skin region r_skin = 2*n*d (d = mcmc step size):
        r_skin_LJ = 2*n_reuse_nblist*np.linalg.norm(width*np.ones(3))
        
        box.compute_LJneighbourlist(r_cut_LJ, r_skin_LJ) # compute neighbourlist for box
        box.compute_LJ_potential(r_cut_LJ, r_skin_LJ) # compute LJ potential for box
    else: # or use provided Box object
        # Set size of r_cut_LJ = 2.5*sigma
        r_cut_LJ = 2.5*np.mean(box.sigmas)
        if width is None: # if no width (~MCMC step size) specified
            width = box.size[0]/5000

        # Set size of skin region r_skin = 2*n*d (d = mcmc step size):
        r_skin_LJ = 2*n_reuse_nblist*np.linalg.norm(width*np.ones(3))

        if box.LJneighbourlists is None:
            box.compute_LJneighbourlist(r_cut_LJ, r_skin_LJ) # compute neighbourlist for box (in case not already computed)
            box.compute_LJ_potential(r_cut_LJ, r_skin_LJ) # compute LJ potential for box using neighbourlist

    return box, r_cut_LJ, r_skin_LJ, width

def create_particle_object(parameters, positions, types):
    particles = []

    for i in range(len(types)):
        position = positions[i]
        sigmaLJ = parameters[types[i]][0]
        epsilonLJ = parameters[types[i]][1]
        charge = parameters[types[i]][3]

        particles.append(system.Particle(position, charge, sigmaLJ, epsilonLJ))

    return particles
