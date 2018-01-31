import unittest
import numpy as np
import lennardjones
import neighbourlist
import numpy.testing as npt

r_cut = 2.5
r_skin = 1.3
positions = np.array([[ 2.90956646, 2.15152179, 3.65165442],[ 3.68121774, 0.18314957, 3.73834638],[ 1.14410085, 4.85833454, 2.95269129],[ 0.78264966, 3.65091237, 1.03486635]])
neighbour_list = neighbourlist.create_verlet_list(positions, r_cut, r_skin)


class TestLennardJones(unittest.TestCase):

	def test_lennard_jones_ij(self):
		r = np.linalg.norm(positions[1] - positions[0])
		self.assertEqual(lennardjones.lennard_jones_ij(r, r_cut), -0.044065854914534311)

	def test_lennard_jones_ij2(self):
		r = np.linalg.norm(positions[0] - positions[3])
		self.assertEqual(lennardjones.lennard_jones_ij(r, r_cut), 0.0)

	def test_calculate_potential(self):
		self.assertEqual(lennardjones.calculate_potential(positions, neighbour_list), -0.071261145028207878)

if __name__ == '__main__':
    unittest.main()