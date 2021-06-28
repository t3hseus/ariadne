import faiss
import numpy as np


def get_indexes(hitsdata: dict) -> dict:
	"""Returns array of faiss indexes by station,
	hitsdata - dictionary with keys: 'x', 'y', 'station'
	"""

	xs = np.expand_dims(hitsdata['x'], axis=1)
	ys = np.expand_dims(hitsdata['y'], axis=1)
	st = np.expand_dims(hitsdata['station'], axis=1)
	hits = np.concatenate((xs, ys, st), axis=1)

	indexes = dict()

	for station in np.unique(hits[:, 2]):
		indexes[station] = faiss.IndexFlatL2(2)
		indexes[station].add(hits[:, :2][hits[:, 2] == station])
	return indexes


def get_nearest_hits(centers: np.array,
                     index: faiss.IndexFlatL2,
                     k_neighbours: int) -> np.array:
	"""Returns k nearest neighbours
	from specified index for each center"""

	d, ids = index.search(centers, k_neighbours)
	all_hits = np.array([index.reconstruct(int(i)) for i in ids.flatten()])
	#maybe need to use array of hits on station and build index right here

	return all_hits


def check_hits_in_ellipse(ellipses: np.array,
                          hits: np.array) -> np.array:
	"""Returns mask for specified hits,
	True if hit in ellipse, False otherwise
	"""

	k_hits = len(hits) / len(ellipses)
	duplicated_ellipses = np.repeat(ellipses, k_hits, axis=0)

	centers = duplicated_ellipses[:, :2]
	r1r2 = duplicated_ellipses[:, 2:]

	tmp = (hits - centers) / r1r2
	tmp = tmp**2

	sums = tmp.sum(axis=1)

	return sums <= 1


def get_new_seeds(seeds: np.array,
                  new_hits: np.array,
                  mask: np.array) -> np.array:
	"""Returns seeds extended with passed hits by mask
	"""
	# maybe need to rewrite without copying all seeds
	k_hits = len(new_hits) / len(seeds)
	duplicated_seeds = np.repeat(seeds, k_hits, axis=0)

	trunc_seeds = duplicated_seeds[mask]
	trunc_hits = new_hits[mask]

	padded_seeds = np.pad(trunc_seeds, ((0, 0), (0, 1), (0, 0)))

	padded_seeds[:, -1, :] = trunc_hits

	return padded_seeds


if __name__ == '__main__':

	# some numbers for test
	hits = {'x': np.linspace(1, 10, 100).astype('float32'),
            'y': np.linspace(1, 10, 100).astype('float32'),
            'station': np.repeat(np.arange(10).astype('float32'), 10)}
	seeds = np.array([[[1, 1], [2, 2]], [[3, 3], [4, 4]]], dtype='float32')
	ellipses = np.array([[1, 1, 0.2, 0.2], [2, 2, 0.1, 0.1]]).astype('float32')

	print('current seeds:')
	print(seeds)
	print()

	indexes = get_indexes(hits)

	# without copy error 'array is not C-contiguous'
	hits = get_nearest_hits(ellipses[:, :2].copy(order='C'), indexes[0], 3)
	print('3 nearest hits for every ellipse in a row:')
	print(hits)
	print()

	# only first 2 hits because r1r2 for second ellipse too small
	mask = check_hits_in_ellipse(ellipses, hits)
	print('mask for all hits:')
	print(mask)
	print()

	new_seeds = get_new_seeds(seeds, hits, mask)
	print('new seeds')
	print(new_seeds)
