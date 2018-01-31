import numpy as np

def lennard_jones_ij(r, r_cut, sigma1=1, epsilon1=1, sigma2=1, epsilon2=1):
    sigma12 = (sigma1 + sigma2)/2
    epsilon12 = np.sqrt(epsilon1 * epsilon2)

    if r >= r_cut:
        return 0.0 #not sure in this part, because neighbourlist

    q = (sigma12 / r)**6
    q_cut = (sigma12 / r_cut)**6

    return (4.0 * epsilon12 * q * (q-1))
    # return (4.0 * epsilon12 * q * (q-1)) - (4.0 * epsilon12 * q_cut *(q_cut - 1.0))

def calculate_potential(positions, neighbour_list, r_cut=2.5, sigmas=1, epsilons=1):
    index_of_number_of_neighoburs = 0
    i = 0
    potential = 0

    while index_of_number_of_neighoburs < len(neighbour_list):
        number_of_neighoburs = neighbour_list[index_of_number_of_neighoburs]

        for j in range(1, number_of_neighoburs+1):
            index_of_neighbour = neighbour_list[index_of_number_of_neighoburs + j]
            
            r = np.linalg.norm(positions[index_of_neighbour] - positions[i])
            potential += lennard_jones_ij(r, r_cut)

        index_of_number_of_neighoburs += number_of_neighoburs + 1
        i += 1

    return potential
