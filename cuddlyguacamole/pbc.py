import numba as nb

# https://en.wikipedia.org/wiki/Periodic_boundary_conditions#(A)_Restrict_particle_coordinates_to_the_simulation_box

def enforce_pbc_coordinates(position, boxsize): #  Restrict particle coordinates to the simulation box
    for i, length in enumerate(boxsize):
        while position[i] > length/2:
            position[i] -= length
        while position[i] <= -length/2:
            position[i] += length
    return position

@nb.jit(nopython = True)
def enforce_pbc_distance(dist_vec, boxsize): #  implement minimum image convention
    for i, length in enumerate(boxsize):
        while dist_vec[i] > length/2:
            dist_vec[i] -= length
        while dist_vec[i] <= -length/2:
            dist_vec[i] += length
    return dist_vec