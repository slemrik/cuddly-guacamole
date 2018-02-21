import numpy as np
import system

def load_input_file(filename):
	#load the data
	#file name:sodium-chloride-example.npz
	with np.load(filename) as fh:
	    print (fh.keys())

	with np.load(filename) as fh:
	    print (fh['readme'])

	with np.load(filename) as fh:
	    box = fh['box']
	    positions = fh['positions']
	    types = fh['types']

	    parameters = fh['parameters'].item()


