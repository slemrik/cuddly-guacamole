# cuddly-guacamole
## WS 2017/18 Computational Sciences, FU Berlin

Python module that 
1) optimizes the system configuration of a system of charged Lennard-Jones particles 
2) simulates the time evolution of the system at a given temperature

using the Metropolis MCMC algorithm with Coulomb and LJ potential.

### how to use:
1. install package:
got to package folder 
	`pip install .`

2. import package: 
	`import cuddlyguacamole as cd`

3. use methods:
required input fileformat: 'sodium-chloride-example.npz'
-optimalization:
	- `NaCl_box = cd.optimize(filename = 'sodium-chloride-example.npz', to_plot=True)`
	- if you already have the Box object: `NaCl_box = cd.optimize(box = NaCl_box, to_plot=True)` 
		
- simulation: 
	- `NaCl_box = cd.simulate(filename = 'sodium-chloride-example.npz', to_plot = True, plot_rdf = True)`
	- if you already have the Box object: `NaCl_box = cd.simulate(box = NaCl_box, to_plot = True, plot_rdf = True)`

4. test in terminal:
`setup.py test`

### Group members include 

* Eva Bertalan
* Henrik Gjoertz
* Jiasheng Lai
* Pooja Pandey
* Yuanwei Pi
