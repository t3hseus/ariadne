import numpy as np
import unittest as ut
import inference

class Tests(ut.TestCase):

	def setUp(self):
		self.hits = {'x': np.linspace(1, 10, 100).astype('float32'),
		             'y': np.linspace(1, 10, 100).astype('float32'),
		             'station': np.repeat(np.arange(10).astype('float32'), 10)}
		self.seeds = np.array([[[1, 1], [2, 2]], [[3, 3], [4, 4]]], dtype='float32')
		self.ellipses = np.array([[1, 1, 0.2, 0.2],
                            [2, 2, 0.1, 0.1]]).astype('float32')
		self.nearest_hits = np.array([[1., 1.],
                                [1.09090912, 1.09090912],
                                [1.18181813, 1.18181813],
                                [1.81818187, 1.81818187],
                                [1.72727275, 1.72727275],
                                [1.63636363, 1.63636363]], dtype='float32')

	def test_nearest(self):
		output = np.array([[1., 1.],
                     [1.09090912, 1.09090912],
                     [1.18181813, 1.18181813],
                     [1.81818187, 1.81818187],
                     [1.72727275, 1.72727275],
                     [1.63636363, 1.63636363]], dtype='float32')

		indexes = inference.get_indexes(self.hits)
		found_hits = inference.get_nearest_hits(
			self.ellipses[:, :2].copy(order='C'), indexes[0], 3)
		self.assertTrue((found_hits == output).all())

	def test_hits_in_ellipse(self):
		output = np.array([True, True, False, False, False, False])

		mask = inference.check_hits_in_ellipse(self.ellipses, self.nearest_hits)
		self.assertTrue((mask == output).all())

	def test_empty_mask(self):
		output = np.array([False, False, False, False, False, False])

		ellipses = np.array([[0.5, 0.5, 0.2, 0.2],
                       [2, 2, 0.1, 0.1]]).astype('float32')

		mask = inference.check_hits_in_ellipse(ellipses, self.nearest_hits)
		self.assertTrue((mask == output).all())

	def test_get_seeds(self):
		output = np.array([[[1., 1.],
                      [2., 2.],
                      [1., 1.]],
                     [[1., 1.],
                      [2., 2.],
                      [1.0909091, 1.0909091]]], dtype='float32')

		mask = np.array([True, True, False, False, False, False])

		new_seeds = inference.get_new_seeds(self.seeds, self.nearest_hits, mask)
		self.assertTrue((new_seeds == output).all())

	def test_empty_seeds(self):
		output = np.empty((0, 3, 2))

		mask = np.array([False, False, False, False, False, False])

		new_seeds = inference.get_new_seeds(self.seeds, self.nearest_hits, mask)
		self.assertTrue((new_seeds == output).all())

	def test_all_functions(self):
		output = np.array([[[1., 1.],
                      [2., 2.],
                      [1., 1.]],
                     [[1., 1.],
                      [2., 2.],
                      [1.0909091, 1.0909091]]], dtype='float32')

		indexes = inference.get_indexes(self.hits)
		hits = inference.get_nearest_hits(
			self.ellipses[:, :2].copy(order='C'), indexes[0], 3)
		mask = inference.check_hits_in_ellipse(self.ellipses, hits)
		new_seeds = inference.get_new_seeds(self.seeds, hits, mask)
		self.assertTrue((new_seeds == output).all())


if __name__ == '__main__':
    ut.main()
