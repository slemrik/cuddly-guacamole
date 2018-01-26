def enforce_pbc(position, boxsize):
    for i, length in enumerate(boxsize):
        while position[i] > length:
            position[i] -= 2*length
        while position[i] <= -length:
            position[i] += 2*length
    return position