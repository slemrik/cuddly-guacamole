def enforce_pbc(position, boxsize):
    for i, length in enumerate(boxsize):
        while position[i] > length/2:
            position[i] -= length
        while position[i] <= -length/2:
            position[i] += length
    return position