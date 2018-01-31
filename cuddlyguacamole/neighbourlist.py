import numpy as np

def verlet_neighbourlist(positions, r_cut, r_skin):
    neighbor_list = []
    start_of_neighbours = 0

    for i in range(0, len(positions)):
        n_neighbours_of_i = 0
        neighbor_list.append(0) 

        for j in range(i+1, len(positions)):
            d = np.linalg.norm(positions[j]-positions[i])

            if not np.array_equal(i,j) and d <= r_cut + r_skin:
                n_neighbours_of_i += 1
                neighbor_list.append(j) #j is the index of particle at positions
        neighbor_list[start_of_neighbours] = n_neighbours_of_i
        start_of_neighbours = len(neighbor_list)

    return neighbor_list
    