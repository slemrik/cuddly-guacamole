# cuddly-guacamole
## WS 2017/18 Computational Sciences, FU Berlin

Python module that 
1) optimizes the system configuration of a system of charged Lennard-Jones particles 
2) simulates the time evolution of the system at a given temperature

using the Metropolis MCMC algorithm with Coulomb and LJ potential.

### how to use:
install package:

	1. got to package folder `pip install .`
	2. import package: `import cuddlyguacamole as cd`
	3. required input fileformat: 'sodium-chloride-example.npz'
	4. run optimalization: `cd.optimize('sodium-chloride-example.npz')`
	5. run simulation: `cd.simulate('sodium-chloride-example.npz')`

test in terminal:
`setup.py test`

### Group members include 

* Eva Bertalan
* Henrik Gjoertz
* Jiasheng Lai
* Pooja Pandey
* Yuanwei Pi
