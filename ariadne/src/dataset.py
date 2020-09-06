import sys

from ariadne import transformations
from ariadne.tracknet_v2 import dataset

if __name__ == '__main__':
    default_path = '../data/200.csv'
    trnf = transformations.Compose([
        transformations.ToCylindrical(drop_old=True),
        transformations.MinMaxScale(columns=['r', 'phi', 'z']),
        transformations.DropSpinningTracks(),
        transformations.DropFakes()
    ])
    data = dataset.BESDataset(csv_file=default_path, preprocessing=trnf)
    print(data[0])
