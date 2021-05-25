import unittest
import sys
import pandas as pd
import numpy as np
import itertools

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
    ConstraintsNormalize,
BakeStationValues,
DropTracksWithHoles
)

path = '../data/200.csv'
path_radial = '../data/200_radial.csv'


import logging
logging.basicConfig(stream=sys.stderr)
logging.getLogger("TestLogger").setLevel(logging.DEBUG)

class StandardTestCase(unittest.TestCase):
    def setUp(self):
        self.data = pd.read_csv(path)
        self.radial_df = pd.read_csv(path_radial)

    def _init_scaler(self, drop_old=True):
        self.scaler = StandardScale(drop_old=drop_old, with_mean=True, with_std=True)

    def test_init(self):
        self._init_scaler()
        self.assertEqual(self.scaler.drop_old, True)
        self._init_scaler(drop_old=False)
        self.assertEqual(self.scaler.drop_old, False)

    def test_transform(self):
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
        temp = self.data.loc[0,'x']
        self.assertEqual(self.scaler(self.data).loc[0, 'x_old'], temp)

class MinMaxTestCase(unittest.TestCase):

    def setUp(self):
        self.data = pd.read_csv(path)
        self.radial_df = pd.read_csv(path_radial)

    def _init_scaler(self, drop_old=True, range=(0,1)):
        self.scaler = MinMaxScale(drop_old=drop_old, feature_range=range)

    def test_init(self):
        self._init_scaler()
        self.assertEqual(self.scaler.drop_old, True)
        self._init_scaler(drop_old=False)
        self.assertEqual(self.scaler.drop_old, False)

    def test_transform_zero_one(self):
        self._init_scaler(drop_old=True, range=(0, 1))
        self.assertEqual(min(self.scaler(self.data).loc[:, 'x']), 0)
        self.assertEqual(max(self.scaler(self.data).loc[:, 'x']), 1)

    def test_transform_minus_one_one(self):
        self._init_scaler(drop_old=True, range=(-1, 1))
        self.assertEqual(min(self.scaler(self.data).loc[:, 'x']), -1)
        self.assertEqual(max(self.scaler(self.data).loc[:, 'x']), 1)

    # def test_drop_true(self):
    #     self._init_scaler(True)
    #     self._init_data()
    #     self.assertRaises(self.scaler(self.data, columns=['x', 'y', 'z']).loc[0, 'x_old'], KeyError)

    def test_drop_false(self):
        self._init_scaler(False)
        temp = self.data.loc[0, 'x']
        self.assertEqual(self.scaler.drop_old, False)
        self.assertEqual(self.scaler(self.data).loc[0, 'x_old'], temp)

class NormalTestCase(unittest.TestCase):

    def _init_scaler(self, drop_old=True, norm='l2'):
        self.scaler = Normalize(drop_old=drop_old, norm=norm)

    def setUp(self):
        self.data = pd.read_csv(path)
        self.radial_df = pd.read_csv(path_radial)

    def test_init(self):
        self._init_scaler()
        self.assertEqual(self.scaler.drop_old, True)
        self._init_scaler(drop_old=False)
        self.assertEqual(self.scaler.drop_old, False)

    def test_transform_l2(self):
        self._init_scaler(drop_old=True)
        self.assertEqual(round(min(self.scaler(self.data).loc[:, 'x']), 2), -1)
        self.assertEqual(round(max(self.scaler(self.data).loc[:, 'x']), 2), 1)

    # def test_drop_true(self):
    #     self._init_scaler(True)
    #     self._init_data()
    #     self.assertRaises(self.scaler(self.data, columns=['x', 'y', 'z']).loc[0, 'x_old'], KeyError)

    def test_drop_false(self):
        self._init_scaler(False)
        temp = self.data.loc[0, 'x']
        self.assertEqual(self.scaler.drop_old, False)
        self.assertEqual(self.scaler(self.data).loc[0, 'x_old'], temp)

class CylindricalTestCase(unittest.TestCase):

    def _init_transformer(self, drop_old=True):
        self.transformer = ToCylindrical(drop_old=drop_old)

    def setUp(self):
        self.data = pd.read_csv(path)
        self.radial_df = pd.read_csv(path_radial)

    def test_init(self):
        self._init_transformer()
        self.assertEqual(self.transformer.drop_old, True)
        self._init_transformer(drop_old=False)
        self.assertEqual(self.transformer.drop_old, False)

    def test_drop_false(self):
        self._init_transformer(False)
        self.assertEqual(self.transformer.drop_old, False)
        self.assertEqual(self.transformer(self.data).loc[0, 'x'], self.data.loc[0, 'x'])

class CartesianTestCase(unittest.TestCase):

    def _init_transformer(self, drop_old=True):
        self.transformer = ToCartesian(drop_old=drop_old)

    def setUp(self):
        self.data = pd.read_csv(path)
        self.radial_df = pd.read_csv(path_radial)

    def test_init(self):
        self._init_transformer()
        self.assertEqual(self.transformer.drop_old, True)
        self._init_transformer(drop_old=False)
        self.assertEqual(self.transformer.drop_old, False)


    def test_transform(self):
        self._init_transformer()
        # this test is working only if data is scaled previosly
        #self.assertEqual(self.transformer(self.data, columns=['x','y','z']).loc[0, 'r'], self.radial_df.loc[0,'r'] )
        #self.assertEqual(self.transformer(self.data, columns=['x', 'y', 'z']).loc[0, 'phi'], self.radial_df.loc[0, 'phi'])
        self.assertAlmostEqual(self.transformer(self.radial_df).loc[0, 'z'], self.radial_df.loc[0, 'z'])
        self.assertAlmostEqual(self.transformer(self.radial_df).loc[0, 'y'], self.radial_df.loc[0, 'y'])
        self.assertAlmostEqual(self.transformer(self.radial_df).loc[0, 'x'], self.radial_df.loc[0, 'x'])

    def test_drop_false(self):
        self._init_transformer(False)
        self.assertEqual(self.transformer(self.radial_df).loc[0, 'phi'], self.radial_df.loc[0,'phi'])

class DropShortTestCase(unittest.TestCase):

    def _init_transformer(self, num_stations=None, keep_filtered=True):
        self.transformer = DropShort(num_stations=num_stations, keep_filtered=keep_filtered)

    def setUp(self):
        self.data = pd.DataFrame({'r': [1., 0.5, 0.1, 0.2, 0.8, 0.6, 0.2],
                                  'phi': [3., 0.5, 2., 0.2, 1.1, -0.5, -0.1],
                                  'z': [0.1, 0.2, 0.33, 0.1, 0.2, 0.2, 0.1],
                                  'track': [1, 1, 1, 2, 2, -1, -1],
                                  'station': [1, 2, 3, 1, 2, 1, 3],
                                  'event': [0, 0, 0, 0, 0, 0, 0]})



    def test_init(self):
        self._init_transformer()
        self.assertEqual(len(self.data), 7)
        self.assertEqual(len(self.data.columns), 6)
        self.assertEqual(self.transformer.num_stations, None)
        self._init_transformer(num_stations=3)
        self.assertEqual(self.transformer.num_stations, 3)
        self.assertEqual(self.transformer._broken_tracks, None)
        self.assertEqual(self.transformer._num_broken_tracks, None)

    def test_transform(self):
        self._init_transformer()
        self.assertEqual(len(self.transformer(self.data)), len(self.data))

    def test_transform_3(self):
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
        self._init_transformer(num_stations=4)
        result = self.transformer(self.data)
        self.assertEqual(len(result[result['track']==-1]), 7)
        self.assertEqual(len(result), len(self.data))
        self._init_transformer(num_stations=4, keep_filtered=False)
        result = self.transformer(self.data)
        self.assertEqual(len(result[result['track'] == -1]), 2)
        self.assertEqual(len(result), 2)

    def test_transform_track(self):
        self._init_transformer(num_stations=3)
        result = self.transformer(self.data)
        result.reset_index(inplace=True, drop=True)
        self.assertEqual(len(result[result.track==-1]), 4)
        self.assertEqual(result.iloc[0, 3], 1)
        self.assertEqual(result.iloc[1, 3], 1)
        self.assertEqual(result.iloc[3, 3], -1)
        self.assertEqual(result.iloc[4, 3], -1)


    def test_get_broken(self):
        self._init_transformer(3)
        self.assertIsNone(self.transformer.get_num_broken())
        self.transformer(self.data)
        self.assertEqual(1, self.transformer.get_num_broken())
        self.transformer(self.data[:3])
        self.assertEqual(0, self.transformer.get_num_broken())

class DropWarpsTestCase(unittest.TestCase):
    def _init_transformer(self, keep_filtered=True):
        self.transformer = DropSpinningTracks(keep_filtered=keep_filtered)

    def setUp(self):
        self.data = pd.DataFrame({'r': [1., 0.5, 0.1, 0.2, 0.8, 0.6, 0.2],
                                  'phi': [3., 0.5, 2., 0.2, 1.1, -0.5, -0.1],
                                  'z': [0.1, 0.2, 0.33, 0.1, 0.2, 0.2, 0.1],
                                  'track': [1, 1, 1, 2, 2, 2, -1],
                                  'station': [1, 2, 3, 1, 2, 2, 3],
                                  'event': [0, 0, 0, 0, 0, 0, 0]})

    def test_init(self):
        self._init_transformer()
        self.assertEqual(len(self.data), 7)
        self.assertEqual(len(self.data.columns), 6)
        self.assertEqual(self.transformer._broken_tracks, None)
        self.assertEqual(self.transformer._num_broken_tracks, None)

    def test_transform(self):
        self._init_transformer()
        result = self.transformer(self.data)
        self.assertEqual(len(result), len(self.data))
        self.assertEqual(len(result[result['track'] == -1]), 4)
        self.assertEqual(result.iloc[0, 3], 1)
        self.assertEqual(result.iloc[3, 3], -1)
        self.assertEqual(result.iloc[5, 3], -1)

    def test_transform_no_keep(self):
        self._init_transformer(keep_filtered=False)
        self.assertEqual(len(self.transformer(self.data)), 4)
        self._init_transformer(keep_filtered=True)
        self.assertEqual(len(self.transformer(self.data)), 7)

    def test_get_broken(self):
        self._init_transformer(2)
        self.assertEqual(self.transformer.get_num_broken(), None)
        self.transformer(self.data)
        self.assertEqual(self.transformer.get_num_broken(), 1)

class DropHolesTestCase(unittest.TestCase):
    def _init_transformer(self, keep_filtered=True,min_station_num=1):
        self.transformer = DropTracksWithHoles(keep_filtered=keep_filtered, min_station_num=min_station_num)

    def setUp(self):
        self.data = pd.DataFrame({'r': [1.,  0.1, 0.2, 0.8, 0.6, 0.2],
                                  'phi': [3.,  2., 0.2, 1.1, -0.5, -0.1],
                                  'z': [0.1,  0.33, 0.1, 0.2, 0.2, 0.1],
                                  'track': [1,  1, 2, 2, 2, -1],
                                  'station': [1, 3, 1, 2, 3, 3],
                                  'event': [0, 0, 0, 0, 0, 0]})

    def test_init(self):
        self._init_transformer()
        self.assertEqual(len(self.data), 6)
        self.assertEqual(len(self.data.columns), 6)
        self.assertEqual(self.transformer._broken_tracks, None)
        self.assertEqual(self.transformer._num_broken_tracks, None)

    def test_transform(self):
        self._init_transformer()
        result = self.transformer(self.data)
        log = logging.getLogger('test')
        log.info(result)
        self.assertEqual(len(result), len(self.data))
        self.assertEqual(len(result[result['track'] == -1]), 3)

    def test_transform_no_keep(self):
        self._init_transformer(keep_filtered=False)
        self.assertEqual(len(self.transformer(self.data)), 4)
        self._init_transformer(keep_filtered=True)
        self.assertEqual(len(self.transformer(self.data)), 6)

    def test_transform_from_zero(self):
        self._init_transformer(keep_filtered=False, min_station_num=0)
        self.assertEqual(len(self.transformer(self.data)), 1)
        self._init_transformer(keep_filtered=True, min_station_num=0)
        self.assertEqual(len(self.transformer(self.data)), 6)

    def test_get_broken(self):
        self._init_transformer(2)
        self.assertEqual(self.transformer.get_num_broken(), None)
        self.transformer(self.data)
        self.assertEqual(self.transformer.get_num_broken(), 1)


class BakeColumnTestCase(unittest.TestCase):
    def _init_transformer(self, keep_filtered=True):
        self.transformer = BakeStationValues(values={0: 0.1, 1:0.3, 2: 0.5, 3:0.7})

    def setUp(self):
        self.data = pd.DataFrame({'r': [1.,  0.1, 0.2, 0.8, 0.6, 0.2],
                                  'phi': [3.,  2., 0.2, 1.1, -0.5, -0.1],
                                  'z': [0.1,  0.33, 0.1, 0.2, 0.2, 0.1],
                                  'track': [1,  1, 2, 2, 2, -1],
                                  'station': [1, 3, 1, 2, 3, 3],
                                  'event': [0, 0, 0, 0, 0, 0]})

    def test_init(self):
        self._init_transformer()
        self.assertEqual(len(self.data), 6)
        self.assertEqual(len(self.data.columns), 6)

    def test_transform(self):
        self._init_transformer()
        result = self.transformer(self.data)
        self.assertEqual(len(result), len(self.data))
        self.assertEqual(len(result[result['z'] == 0.1]), 0)
        self.assertEqual(len(result[result['z'] == 0.3]), 2)
        self.assertEqual(len(result[result['z'] == 0.5]), 1)
        self.assertEqual(len(result[result['z'] == 0.7]), 3)


class ComposeTestCase(unittest.TestCase):
    def _init_transformer(self):
        self.transformer = Compose([
            StandardScale(),
            ToCylindrical()
            ]
        )

    def setUp(self):
        self.data = pd.read_csv(path)
        self.radial_df = pd.read_csv(path_radial)

    def test_init(self):
        self._init_transformer()

    def test_transform(self):
        self._init_transformer()
        self.assertEqual(len(self.transformer(self.data)), 100000)

    def test_coords(self):
        self._init_transformer()
        transformed = self.transformer(self.data)
        self.assertAlmostEqual(transformed.loc[0, 'z'], self.radial_df.loc[0, 'z'])
        self.assertAlmostEqual(transformed.loc[0, 'y'], self.radial_df.loc[0, 'y'])
        self.assertAlmostEqual(transformed.loc[0, 'x'], self.radial_df.loc[0, 'x'])

    def test_polar(self):
        self._init_transformer()
        transformed = self.transformer(self.data)
        self.assertAlmostEqual(transformed.loc[0, 'phi'], self.radial_df.loc[0, 'phi'])
        self.assertAlmostEqual(transformed.loc[0, 'r'], self.radial_df.loc[0, 'r'])


class ToBucketsTestCase(unittest.TestCase):

    def _init_transformer(self, flat=True, shuffle=False, max_stations=None, max_bucket_size=None, keep_fakes=True):
        self.transformer = ToBuckets(flat=flat, shuffle=shuffle, max_stations=max_stations, max_bucket_size=max_bucket_size, keep_fakes=keep_fakes)

    def setUp(self):
        self.data = pd.DataFrame({'r': [1., 0.5, 0.1, 0.2, 0.1, 0.2, 0.8, 0.6, 0.2,0.8, 0.6, 0.2,0.2, 0.2, 0.1,0.2, 0.1,0.2, 0.2, 0.1],
                                  'phi': [3., 0.5, 2., 0.2, 1.1, -0.5, -0.1, 2., 0.2, 1.1, -0.5, -0.1, 0.2, 0.2, 0.1,0.2, 0.1,0.2, 0.2, 0.1],
                                  'z': [0.1, 0.2, 0.33, 0.1, 0.2, 0.2, 0.1, 0.33, 0.1, 0.2, 0.2, 0.1,0.2, 0.2, 0.1,0.2, 0.1,0.2, 0.2, 0.1],
                                  'track': [1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 5],
                                  'station': [1, 2, 3, 4, 5, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 5],
                                  'event': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0,0, 0, 0, 0, 0]})

    def test_init(self):
        self._init_transformer()
        self.assertEqual(len(self.data), 20)
        self.assertEqual(len(self.data.columns), 6)
        self.assertEqual(self.transformer.max_num_stations, None)
        self._init_transformer(shuffle=True, max_stations=3)
        self.assertEqual(self.transformer.max_num_stations, 3)

    def test_transform(self):
        log = logging.getLogger("TestLogger")
        self._init_transformer()
        result = self.transformer(self.data)
        self.assertEqual(len(result), 20)
        self.assertEqual(len(result[result['bucket'] == 5]), 10)
        self.assertEqual(len(result[result['bucket'] == 4]), 4)
        self.assertEqual(len(result[result['bucket'] == 3]), 6)
        self.assertEqual(self.transformer.max_num_stations, 5)

    def test_transform_not_flat(self):
        self._init_transformer(flat=False)
        result = self.transformer(self.data)
        self.assertEqual(len(result[5]), 10)
        self.assertEqual(len(result[4]), 4)
        self.assertEqual(len(result[3]), 6)

    def test_transform_with_balancing(self):
        self._init_transformer(flat=False)
        track_list = []
        station_list = []
        event_list = []
        num_tracks = [2, 10]
        for num, i in enumerate(range(3, 5)):
            for track in range(num_tracks[num]):
                track_list.append(np.full(i, i*1000+track))
                event_list.append(np.zeros(i))
                station_list.append(np.arange(1, i+1))
        events = list(np.concatenate(event_list))
        self.data = pd.DataFrame(
            {
             'track': list(np.concatenate(track_list)),
             'station': list(np.concatenate(station_list)),
             'event': events,
             'r': list(np.random.rand(len(events))),
             'phi': list(np.random.rand(len(events))),
             'z': list(np.random.rand(len(events))),
             })
        result = self.transformer(self.data)
        #log.info(f'\n {result}')
        #log.info(np.in1d(result[3], result[4]))
        #log.info(result[4])
        #log.info(result[3])
        self.assertEqual(any(np.in1d(result[3]['index'], result[4]['index'])), False)
        self.assertEqual(len(result[4]), 10*4)
        self.assertEqual(len(result[3]), 2*3)

    def test_transform_with_balancing_and_bucket_size(self):
        self._init_transformer(flat=False, max_bucket_size=6)
        track_list = []
        station_list = []
        event_list = []
        num_tracks = [2, 10]
        for num, i in enumerate(range(3, 5)):
            for track in range(num_tracks[num]):
                track_list.append(np.full(i, i*1000+track))
                event_list.append(np.zeros(i))
                station_list.append(np.arange(1, i+1))
        events = list(np.concatenate(event_list))
        self.data = pd.DataFrame(
            {
             'track': list(np.concatenate(track_list)),
             'station': list(np.concatenate(station_list)),
             'event': events,
             'r': list(np.random.rand(len(events))),
             'phi': list(np.random.rand(len(events))),
             'z': list(np.random.rand(len(events))),
             })
        log = logging.getLogger("TestLogger")
        result = self.transformer(self.data)
        #log.info(f'\n {result}')
        #log.info(np.in1d(result[3], result[4]))
        #log.info(result[4])
        #log.info(result[3])
        self.assertEqual(any(np.in1d(result[3]['index'], result[4]['index'])), False)
        self.assertEqual(len(result[4]), 6*4)
        self.assertEqual(len(result[3]), 6*3)

    def test_transform_with_empty_bucket_and_bucket_size(self):
        self._init_transformer(flat=False, max_bucket_size=6)
        track_list = []
        station_list = []
        event_list = []
        num_tracks = [2, 0, 10]
        for num, i in enumerate(range(3, 6)):
            if i == 1:
                continue
            for track in range(num_tracks[num]):
                track_list.append(np.full(i, i*1000+track))
                event_list.append(np.zeros(i))
                station_list.append(np.arange(1, i+1))
        events = list(np.concatenate(event_list))
        self.data = pd.DataFrame(
            {
             'track': list(np.concatenate(track_list)),
             'station': list(np.concatenate(station_list)),
             'event': events,
             'r': list(np.random.rand(len(events))),
             'phi': list(np.random.rand(len(events))),
             'z': list(np.random.rand(len(events))),
             })
        log = logging.getLogger("TestLogger")
        result = self.transformer(self.data)
        #log.info(f'\n {result}')
        #log.info(np.in1d(result[3], result[4]))
        #log.info(result[4])
        #log.info(result[3])
        self.assertEqual(any(np.in1d(result[3]['index'], result[5]['index'])), False)
        self.assertEqual(len(result[5]), 6*5)
        self.assertEqual(len(result[4]), 4*4)
        self.assertEqual(len(result[3]), 2*3)

    def test_transform_with_longer_bucket_and_bucket_size_and_maxlen(self):
        self._init_transformer(flat=False, max_bucket_size=6, max_stations=4)
        track_list = []
        station_list = []
        event_list = []
        num_tracks = [2, 3, 10]
        for num, i in enumerate(range(3, 6)):
            for track in range(num_tracks[num]):
                track_list.append(np.full(i, i*1000+track))
                event_list.append(np.zeros(i))
                station_list.append(np.arange(1, i+1))
        events = list(np.concatenate(event_list))
        self.data = pd.DataFrame(
            {
             'track': list(np.concatenate(track_list)),
             'station': list(np.concatenate(station_list)),
             'event': events,
             'r': list(np.random.rand(len(events))),
             'phi': list(np.random.rand(len(events))),
             'z': list(np.random.rand(len(events))),
             })
        log = logging.getLogger("TestLogger")
        result = self.transformer(self.data)
        #log.info(f'\n {result}')
        #log.info(np.in1d(result[3], result[4]))
        #log.info(result[4])
        #log.info(result[3])
        self.assertEqual(any(np.in1d(result[3]['index'], result[4]['index'])), False)
        self.assertEqual(len(result[3]), 6*3)
        self.assertEqual(len(result[4]), 6*4)
        self.assertEqual(5 in result.keys(), False)

    def test_transform_with_longer_bucket_and_bucket_size_and_maxlen_flat(self):
        self._init_transformer(flat=True, max_bucket_size=6, max_stations=4)
        track_list = []
        station_list = []
        event_list = []
        num_tracks = [2, 3, 10]
        for num, i in enumerate(range(3, 6)):
            for track in range(num_tracks[num]):
                track_list.append(np.full(i, i*1000+track))
                event_list.append(np.zeros(i))
                station_list.append(np.arange(1, i+1))
        events = list(np.concatenate(event_list))
        self.data = pd.DataFrame(
            {
             'track': list(np.concatenate(track_list)),
             'station': list(np.concatenate(station_list)),
             'event': events,
             'r': list(np.random.rand(len(events))),
             'phi': list(np.random.rand(len(events))),
             'z': list(np.random.rand(len(events))),
             })
        log = logging.getLogger("TestLogger")
        result = self.transformer(self.data)
        #log.info(f'\n {result}')
        #log.info(np.in1d(result[3], result[4]))
        #log.info(result[4])
        #log.info(result[3])
        self.assertEqual(any(np.in1d(result[result['bucket']==3]['index'], result[result['bucket']==4]['index'])), False)
        self.assertEqual(len(result[result['bucket']==3]), 6*3)
        self.assertEqual(len(result[result['bucket']==4]), 6*4)
        self.assertEqual(5 in result['bucket'], False)

    def test_transform_bes(self):
        data = pd.DataFrame({'r': [1., 0.5, 0.1, 0.2, 0.8, 0.6],
                              'phi': [3., 0.5, 2., 0.2, 1.1, -0.5],
                              'z': [0.1, 0.2, 0.33, 0.1, 0.2, 0.2],
                              'track': [1, 1, 1, 2, 2, 2],
                              'station': [1, 2, 3, 1, 2, 2],
                              'event': [0, 0, 0, 0, 0, 0]})
        self._init_transformer(flat=False, keep_fakes=False)
        result = self.transformer(data)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[3]), 2*3)
        self._init_transformer(flat=True)
        result = self.transformer(data)
        self.assertEqual(len(result), 6)
        self.assertEqual(all(result['bucket'] == 3), True)

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
