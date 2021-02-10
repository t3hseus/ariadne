import numpy as np
import pandas as pd
from ariadne import preprocessing

path = '../data/200.csv'
output_path = '../data/data_preprocessed.csv'
def preprocess(path, trnf, output_path):

    if path.split('.')[-1]=='txt':
        data = pd.read_table(path, sep=',', columns=['event', 'x', 'y', 'z', 'station', 'track'], header=None)
    elif path.split('.')[-1]=='csv':
        data = pd.read_csv(path, index_col=None)
    else:
        raise ValueError(f"This format is not supported yet: {path.split('.')[-1]}. Try one of (csv,txt)")
    assert len(data.columns)==6, 'Data must contain 6 columns with coordinates and info about event, station, track'

    data = trnf(data)
    data.to_csv(output_path, index=False)

if __name__ == '__main__':

    trnf = preprocessing.Compose([
        preprocessing.ToCylindrical(drop_old=False),
        preprocessing.MinMaxScale(columns=['r', 'phi', 'z'])
    ])

    preprocess(path=path, trnf=trnf, output_path=output_path)