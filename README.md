# cuddly-guacamole
## WS 2017/18 Computational Sciences, FU Berlin

Python module that 
1) optimizes the system configuration of a system of charged Lennard-Jones particles 
2) simulates the time evolution of the system at a given temperature

using the Metropolis MCMC algorithm with Coulomb and LJ potential.

### how to use:
install package:
	got to package folder 
		`pip install .`
	import package: 
		`import cuddlyguacamole as cd`
	required input fileformat: 'sodium-chloride-example.npz'
	run optimalization: 
		`cd.optimize('sodium-chloride-example.npz')`
	run simulation: 
		`cd.simulate('sodium-chloride-example.npz')`

test in terminal:
`setup.py test`

### Group members include 

* Eva Bertalan
* Henrik Gjoertz
* Jiasheng Lai
* Pooja Pandey
* Yuanwei Pi
