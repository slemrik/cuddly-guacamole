import numpy as np
import system

def load_input_file(filename):

	with np.load(filename) as fh:
	    print (fh.keys())

	with np.load(filename) as fh:
	    print (fh['readme'])

	with np.load(filename) as fh:
	    box = fh['box']
	    positions = fh['positions']
	    types = fh['types']

	    parameters = fh['parameters'].item()

	return parameters, positions, types, box

# load_input_file('sodium-chloride-example.npz')
#remove this later, it will come from function call


