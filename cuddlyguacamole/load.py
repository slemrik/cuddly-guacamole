import numpy as np
import system


#load the data
#file name:sodium-chloride-example.npz
with np.load('sodium-chloride-example.npz') as fh:
    print (fh.keys())

with np.load('sodium-chloride-example.npz') as fh:
    print (fh['readme'])

with np.load('sodium-chloride-example.npz') as fh:
    box = fh['box']
    positions = fh['positions']
    types = fh['types']

    parameters = fh['parameters'].item()

print("\rData: types:[sigma_LJ, epsilon_LJ, mass, charge]")     
print(parameters) 
print("Number of particles :",len(types)) #128 particles
print("The box length is:", box)
#Types index: 64 Na+ & 64 Cl-
#print(types)
#Positions index: each particle's position in [[x1,y1,z1],[x2,y2,z2]...]
#print(positions)




