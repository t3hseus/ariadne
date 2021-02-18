import unittest
import sys
import pandas as pd

from ariadne.transformations import (
    StandardScale,
    MinMaxScale,
    ToCylindrical,
    Normalize, 
    DropShort, 
    DropSpinningTracks, 
    ToCartesian,
    Compose, 
    ToBuckets, 
    ConstraintsNormalize
)

path = '../data/200.csv'
path_radial = '../data/200_radial.csv'



class StandardTestCase(unittest.TestCase):

    def _init_scaler(self, drop_old=True):
        self.scaler = StandardScale(drop_old=drop_old, with_mean=True, with_std=True)

    def _init_data(self):
        self.data = pd.read_csv(path)
        self.radial_df = pd.read_csv(path_radial)

    def test_init(self):
        self._init_scaler()
        self._init_data()
        self.assertEqual(self.scaler.drop_old, True)
        self._init_scaler(drop_old=False)
        self.assertEqual(self.scaler.drop_old, False)


    def test_transform(self):
        self._init_data()
        self._init_scaler()
        temp = self.radial_df.loc[0, :]
        self.assertAlmostEqual(self.scaler(self.data).loc[0, 'x'], temp['x'])
        self.assertAlmostEqual(self.scaler(self.data).loc[0, 'y'], temp['y'])
        self.assertAlmostEqual(self.scaler(self.data).loc[0, 'z'], temp['z'])
    #
    # def test_drop_true(self):
    #     self._init_data()
    #     self._init_scaler(True)
    #     self.assertRaises(self.scaler(self.data, columns=['x', 'y', 'z']).loc[0, 'x_old'], KeyError)

    def test_drop_false(self):
        self._init_scaler(drop_old=False)
        self._init_data()
        temp = self.data.loc[0,'x']
        self.assertEqual(self.scaler(self.data).loc[0, 'x_old'], temp)

class MinMaxTestCase(unittest.TestCase):

    def _init_scaler(self, drop_old=True, range=(0,1)):
        self.scaler = MinMaxScale(drop_old=drop_old, feature_range=range)

    def _init_data(self):
        self.data = pd.read_csv(path)
        self.radial_df = pd.read_csv(path_radial)

    def test_init(self):
        self._init_data()
        self._init_scaler()
        self.assertEqual(self.scaler.drop_old, True)
        self._init_scaler(drop_old=False)
        self.assertEqual(self.scaler.drop_old, False)

    def test_transform_zero_one(self):
        self._init_scaler(drop_old=True, range=(0, 1))
        self._init_data()
        self.assertEqual(min(self.scaler(self.data).loc[:, 'x']), 0)
        self.assertEqual(max(self.scaler(self.data).loc[:, 'x']), 1)

    def test_transform_minus_one_one(self):
        self._init_scaler(drop_old=True, range=(-1, 1))
        self._init_data()
        self.assertEqual(min(self.scaler(self.data).loc[:, 'x']), -1)
        self.assertEqual(max(self.scaler(self.data).loc[:, 'x']), 1)

    # def test_drop_true(self):
    #     self._init_scaler(True)
    #     self._init_data()
    #     self.assertRaises(self.scaler(self.data, columns=['x', 'y', 'z']).loc[0, 'x_old'], KeyError)

    def test_drop_false(self):
        self._init_scaler(False)
        self._init_data()
        temp = self.data.loc[0, 'x']
        self.assertEqual(self.scaler.drop_old, False)
        self.assertEqual(self.scaler(self.data).loc[0, 'x_old'], temp)

class NormalTestCase(unittest.TestCase):

    def _init_scaler(self, drop_old=True, norm='l2'):
        self.scaler = Normalize(drop_old=drop_old, norm=norm)

    def _init_data(self):
        self.data = pd.read_csv(path)
        self.radial_df = pd.read_csv(path_radial)

    def test_init(self):
        self._init_data()
        self._init_scaler()
        self.assertEqual(self.scaler.drop_old, True)
        self._init_scaler(drop_old=False)
        self.assertEqual(self.scaler.drop_old, False)

    def test_transform_l2(self):
        self._init_scaler(drop_old=True)
        self._init_data()
        self.assertEqual(round(min(self.scaler(self.data).loc[:, 'x']), 2), -1)
        self.assertEqual(round(max(self.scaler(self.data).loc[:, 'x']), 2), 1)

    # def test_drop_true(self):
    #     self._init_scaler(True)
    #     self._init_data()
    #     self.assertRaises(self.scaler(self.data, columns=['x', 'y', 'z']).loc[0, 'x_old'], KeyError)

    def test_drop_false(self):
        self._init_scaler(False)
        self._init_data()
        temp = self.data.loc[0, 'x']
        self.assertEqual(self.scaler.drop_old, False)
        self.assertEqual(self.scaler(self.data).loc[0, 'x_old'], temp)

class CylindricalTestCase(unittest.TestCase):

    def _init_transformer(self, drop_old=True):
        self.transformer = ToCylindrical(drop_old=drop_old)

    def _init_data(self):
        self.data = pd.read_csv(path)
        self.radial_df = pd.read_csv(path_radial)

    def test_init(self):
        self._init_transformer()
        self._init_data()
        self.assertEqual(self.transformer.drop_old, True)
        self._init_transformer(drop_old=False)
        self.assertEqual(self.transformer.drop_old, False)

    def test_drop_false(self):
        self._init_data()
        #self.transformer = ToCylindrical(drop_old=False)

        self._init_transformer(False)
        self.assertEqual(self.transformer.drop_old, False)
        self.assertEqual(self.transformer(self.data).loc[0, 'x'], self.data.loc[0, 'x'])

class CartesianTestCase(unittest.TestCase):

    def _init_transformer(self, drop_old=True):
        self.transformer = ToCartesian(drop_old=drop_old)

    def _init_data(self):
        self.data = pd.read_csv(path)
        self.radial_df = pd.read_csv(path_radial)

    def test_init(self):
        self._init_transformer()
        self._init_data()
        self.assertEqual(self.transformer.drop_old, True)
        self._init_transformer(drop_old=False)
        self.assertEqual(self.transformer.drop_old, False)


    def test_transform(self):
        self._init_data()
        self._init_transformer()
        # this test is working only if data is scaled previosly
        #self.assertEqual(self.transformer(self.data, columns=['x','y','z']).loc[0, 'r'], self.radial_df.loc[0,'r'] )
        #self.assertEqual(self.transformer(self.data, columns=['x', 'y', 'z']).loc[0, 'phi'], self.radial_df.loc[0, 'phi'])
        self.assertAlmostEqual(self.transformer(self.radial_df).loc[0, 'z'], self.radial_df.loc[0, 'z'])
        self.assertAlmostEqual(self.transformer(self.radial_df).loc[0, 'y'], self.radial_df.loc[0, 'y'])
        self.assertAlmostEqual(self.transformer(self.radial_df).loc[0, 'x'], self.radial_df.loc[0, 'x'])

    def test_drop_false(self):
        self._init_transformer(False)
        self._init_data()
        self.assertEqual(self.transformer(self.radial_df).loc[0, 'phi'], self.radial_df.loc[0,'phi'])

class DropShortTestCase(unittest.TestCase):

    def _init_transformer(self, num_stations=None, keep_filtered=True):
        self.transformer = DropShort(num_stations=num_stations, keep_filtered=keep_filtered)

    def _init_data(self):
        self.data = pd.DataFrame({'r': [1., 0.5, 0.1, 0.2, 0.8, 0.6, 0.2],
                     'phi': [3., 0.5, 2., 0.2, 1.1, -0.5, -0.1 ],
                     'z':[0.1, 0.2, 0.33, 0.1, 0.2, 0.2, 0.1],
                     'track':[1, 1, 1, 2, 2, -1, -1],
                     'station': [1, 2, 3, 1, 2, 1, 3],
                     'event':[0, 0, 0, 0, 0, 0, 0]})


    def test_init(self):
        self._init_transformer()
        self._init_data()
        self.assertEqual(len(self.data), 7)
        self.assertEqual(len(self.data.columns), 6)
        self.assertEqual(self.transformer.num_stations, None)
        self._init_transformer(num_stations=3)
        self.assertEqual(self.transformer.num_stations, 3)
        self.assertEqual(self.transformer._broken_tracks, None)
        self.assertEqual(self.transformer._num_broken_tracks, None)

    def test_transform(self):
        self._init_data()
        self._init_transformer()
        self.assertEqual(len(self.transformer(self.data)), len(self.data))

    def test_transform_3(self):
        self._init_data()
        self._init_transformer(3)
        self.assertEqual(len(self.transformer(self.data)), len(self.data))
        self._init_transformer(num_stations=3, keep_filtered=True)
        result = self.transformer(self.data)
        self.assertEqual(len(result[result['track'] == -1]), 4)
        self.assertEqual(len(result), len(self.data))
        self._init_transformer(num_stations=3, keep_filtered=False)
        result = self.transformer(self.data)
        self.assertEqual(len(result[result['track'] == -1]), 2)
        self.assertEqual(len(result), len(self.data) - 2)

    def test_transform_no_keep_4(self):
        self._init_data()
        self._init_transformer(num_stations=4)
        result = self.transformer(self.data)
        self.assertEqual(len(result[result['track']==-1]), 7)
        self.assertEqual(len(result), len(self.data))
        self._init_transformer(num_stations=4, keep_filtered=False)
        result = self.transformer(self.data)
        self.assertEqual(len(result[result['track'] == -1]), 2)
        self.assertEqual(len(result), 2)

    def test_transform_track(self):
        self._init_data()
        self._init_transformer(num_stations=3)
        result = self.transformer(self.data)
        result.reset_index(inplace=True, drop=True)
        self.assertEqual(len(result[result.track==-1]), 4)
        self.assertEqual(result.iloc[0,4], 1)
        self.assertEqual(result.iloc[1,4], 1)
        self.assertEqual(result.iloc[3,4], -1)
        self.assertEqual(result.iloc[4,4], -1)


    def test_get_broken(self):
        self._init_data()
        self._init_transformer(3)
        self.assertIsNone(self.transformer.get_num_broken())
        self.transformer(self.data)
        self.assertEqual(1, self.transformer.get_num_broken())
        self.transformer(self.data[:3])
        self.assertEqual(0, self.transformer.get_num_broken())

class DropWarpsTestCase(unittest.TestCase):
    def _init_transformer(self, keep_filtered=True):
        self.transformer = DropSpinningTracks(keep_filtered=keep_filtered)

    def _init_data(self):
        self.data = pd.DataFrame({'r': [1., 0.5, 0.1, 0.2, 0.8, 0.6, 0.2],
                                  'phi': [3., 0.5, 2., 0.2, 1.1, -0.5, -0.1],
                                  'z': [0.1, 0.2, 0.33, 0.1, 0.2, 0.2, 0.1],
                                  'track': [1, 1, 1, 2, 2, 2, -1],
                                  'station': [1, 2, 3, 1, 2, 2, 3],
                                  'event': [0, 0, 0, 0, 0, 0, 0]})

    def test_init(self):
        self._init_transformer()
        self._init_data()
        self.assertEqual(len(self.data), 7)
        self.assertEqual(len(self.data.columns), 6)
        self.assertEqual(self.transformer._broken_tracks, None)
        self.assertEqual(self.transformer._num_broken_tracks, None)

    def test_transform(self):
        self._init_data()
        self._init_transformer()
        result = self.transformer(self.data)
        self.assertEqual(len(result), len(self.data))
        self.assertEqual(len(result[result['track']==-1]), 4)
        self.assertEqual(result.iloc[0,4], 1)
        self.assertEqual(result.iloc[3, 4], -1)
        self.assertEqual(result.iloc[5, 4], -1)

    def test_transform_no_keep(self):
        self._init_data()
        self._init_transformer(keep_filtered=False)
        self.assertEqual(len(self.transformer(self.data)), 4)
        self._init_transformer(keep_filtered=True)
        self.assertEqual(len(self.transformer(self.data)), 7)

    def test_get_broken(self):
        self._init_data()
        self._init_transformer(2)
        self.assertEqual(self.transformer.get_num_broken(), None)
        self.transformer(self.data)
        self.assertEqual(self.transformer.get_num_broken(), 1)

# class DropMissesTestCase(unittest.TestCase):
#     def _init_transformer(self):
#         self.transformer = DropMisses()

#     def _init_data(self):
#         self.data = pd.read_csv('/home/nastya/tracknet/data/200.csv')
#         self.radial_df = pd.read_csv('/home/nastya/tracknet/data/200_radial.csv')

#     def test_init(self):
#         self._init_transformer()
#         self._init_data()
#         self.assertEqual(self.transformer._num_misses, None)

#     def test_transform(self):
#         self._init_data()
#         self._init_transformer()
#         self.assertEqual(len(self.transformer(self.data)), 100000-76969)

#     def test_get_num_misses(self):
#         self._init_data()
#         self._init_transformer()
#         transformed = self.transformer(self.data)
#         self.assertEqual(self.transformer.get_num_misses(), 76969)

class ComposeTestCase(unittest.TestCase):
    def _init_transformer(self):
        self.transformer = Compose([
            StandardScale(),
            ToCylindrical()
            ]
        )

    def _init_data(self):
        self.data = pd.read_csv(path)
        self.radial_df = pd.read_csv(path_radial)

    def test_init(self):
        self._init_transformer()
        self._init_data()

    def test_transform(self):
        self._init_data()
        self._init_transformer()
        self.assertEqual(len(self.transformer(self.data)), 100000)

    def test_coords(self):
        self._init_data()
        self._init_transformer()
        transformed = self.transformer(self.data)
        self.assertAlmostEqual(transformed.loc[0, 'z'], self.radial_df.loc[0, 'z'])
        self.assertAlmostEqual(transformed.loc[0, 'y'], self.radial_df.loc[0, 'y'])
        self.assertAlmostEqual(transformed.loc[0, 'x'], self.radial_df.loc[0, 'x'])

    def test_polar(self):
        self._init_data()
        self._init_transformer()
        transformed = self.transformer(self.data)
        self.assertAlmostEqual(transformed.loc[0, 'phi'], self.radial_df.loc[0, 'phi'])
        self.assertAlmostEqual(transformed.loc[0, 'r'], self.radial_df.loc[0, 'r'])
'''
class ToBucketsTestCase(unittest.TestCase):
    def _init_transformer(self, flat=True, keep_fakes=True):
        self.transformer = ToBuckets(flat=flat, keep_fakes=keep_fakes)

    def _init_data(self):
        self.data = pd.read_csv(path)
        self.radial_df = pd.read_csv(path_radial)

    def test_init(self):
        self._init_transformer()
        self._init_data()
        self.assertEqual(self.transformer.keep_fakes, True)
        self.assertEqual(self.transformer.flat, True)

    def test_misses(self):
        self._init_data()
        self._init_transformer(keep_fakes=False)
        self.assertEqual(len(self.transformer(self.data)), 100000-76969)
        self._init_transformer(keep_fakes=True)
        self.assertEqual(len(self.transformer(self.data)), 100000)

    def test_flat(self):
        self._init_data()
        self._init_transformer()
        self.assertEqual(len(self.transformer(self.data).head()),5)

    def test_no_flat(self):
        self._init_data()
        self._init_transformer(flat=False)
        self.assertEqual(len(self.transformer(self.data)[3].head()), 5)

    def test_bucket_lens(self):
        self._init_data()
        self._init_transformer(keep_fakes=False)
        transformed = self.transformer(self.data)
        self.assertEqual(23031, self.transformer.get_buckets_sizes()[3])


class ConstraintsTestCase(unittest.TestCase):
    def _init_transformer(self, drop_old=True, columns=('x', 'y', 'z'), use_global_constraints=True):
        self.transformer = ConstraintsNormalize(drop_old=drop_old, columns=columns, use_global_constraints=use_global_constraints)

    def _init_data(self):
        self.data = pd.read_csv(path)
        self.radial_df = pd.read_csv(path_radial)

    def test_init(self):
        self._init_transformer()
        self._init_data()
        self.assertEqual(self.transformer.drop_old, True)
        self.assertEqual(len(self.transformer.columns), 3)
        self.assertEqual(self.transformer.margin, 1e-3)

    def test_values_constraints(self):
        self._init_data()
        self._init_transformer()
        transformed = self.transformer(self.data)
        self.assertLessEqual(max(transformed['x']),1)
        self.assertLessEqual(-1, min(transformed['x']))
        self.assertLessEqual(max(transformed['y']), 1)
        self.assertLessEqual(-1, min(transformed['y']))
        self.assertLessEqual(max(transformed['z']), 1)
        self.assertLessEqual(-1, min(transformed['z']))

    def test_lens(self):
        self._init_data()
        self._init_transformer()
        self.assertEqual( 100000, len(self.transformer(self.data)))
'''

if __name__ == '__main__':
    unittest.main()
