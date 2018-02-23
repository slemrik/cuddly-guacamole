import cuddlyguacamole
import test.generate_test_system
import numpy as np

r_cut_LJ = 2.5
r_skin_LJ = 0
testbox = test.generate_test_system.generate_test_system(3, 15*np.ones(3), 64, 200, 0, 1, 1)
testbox.compute_LJneighbourlist(r_cut_LJ, r_skin_LJ)
testbox.compute_LJ_potential(r_cut_LJ, r_skin_LJ)
testbox = cuddlyguacamole.optimize(box = testbox, to_plot=True)
testbox = cuddlyguacamole.simulate(box = testbox, to_plot = True, plot_rdf = True)

# NaCl_box = cuddlyguacamole.optimize(filename = 'sodium-chloride-example.npz', to_plot=True)

# NaCl_box = cuddlyguacamole.simulate(box = NaCl_box, to_plot = True, plot_rdf = True)