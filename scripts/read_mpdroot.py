import argparse
import glob
import os
import numpy as np

argparser = argparse.ArgumentParser()
argparser.add_argument(
    '-e', '--event',
    type=int,
    help='event id',
    default=0
)

DTYPE = np.dtype([
    ('event', np.uint32),
    ('wtf', np.uint32),
    ('x', np.float64),
    ('y', np.float64),
    ('z', np.float64),
    ('station', np.uint32),
    ('track', np.uint32),
    ('px', np.float64),
    ('py', np.float64),
    ('pz', np.float64),
    ('vx', np.float64),
    ('vy', np.float64),
    ('vz', np.float64)
])

if __name__ == '__main__':
    args = argparser.parse_args()
    data_path = '/eos/eos.jinr.ru/nica/mpd/dirac/mpd.nica.jinr/vo/mpd/data/nn/lustre/stor1/dirac/'
    files = glob.glob(os.path.join(data_path, '*.dat'))
    print(f'File: {files[0]}')
    data = np.fromfile(files[0], dtype=DTYPE)
    print(f'Event: {args.event}')
    event = data[data['event'] == args.event]
    print(f'Number of hits in event: {len(event)}')
    print(f"Number of tracks in event: {np.unique(event['track']).size}")

    '''
    Number of hits in event: 82469
    Number of tracks in event: 704
    '''