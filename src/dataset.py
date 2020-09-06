import sys
sys.path.append('..')
from ariadne import preprocessing
from ariadne.tracknet_v2 import dataset

if __name__ == '__main__':

    default_path = '../data/200.csv'
    trnf = preprocessing.Compose([
        preprocessing.ToCylindrical(drop_old=True),
        preprocessing.MinMaxScale(columns=['r', 'phi', 'z']),
        preprocessing.DropSpinningTracks(),
        preprocessing.DropFakes()
    ])
    data = dataset.BESDataset(csv_file=default_path, preprocessing=trnf)
    print(data[0])