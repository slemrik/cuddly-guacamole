import unittest
import numpy as np
import neighbourlist
import numpy.testing as npt

positions = np.array([[ 2.90956646, 2.15152179, 3.65165442],[ 3.68121774, 0.18314957, 3.73834638],[ 1.14410085, 4.85833454, 2.95269129],[ 0.78264966, 3.65091237, 1.03486635]])

neighbour_list = [3, 1, 2, 3, 0, 1, 3, 0]

r_cut = 2.5
r_skin = 1.3

class TestNeighbourList(unittest.TestCase):

    def test_verlet_list(self):
        npt.assert_array_equal(neighbourlist.create_verlet_list(positions, r_cut, r_skin), neighbour_list)

if __name__ == '__main__':
    unittest.main()