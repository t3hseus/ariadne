import numpy as np
import pandas as pd
import argparse
from ariadne import preprocessing

def preprocess(path, trnf, output_path):

    if path.split('.')[-1]=='txt':
        data = pd.read_table(path, sep=',', columns=['event', 'x', 'y', 'z', 'station', 'track'], header=None)

    elif path.split('.')[-1]=='csv':
        data = pd.read_csv(path, index_col=None)
    else:
        raise ValueError(f"This format is not supported yet: {path.split('.')[-1]}. Try one of (csv,txt)")
    assert len(data.columns) == 6, 'Data must contain 6 columns with coordinates and info about event, station, track'
    data = trnf(data)
    data.to_csv(output_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data preprocessing')
    parser.add_argument('input_file', type=str, default='../data/200.csv',
                        help='Path to data file (csv or txt)')
    parser.add_argument('output_file', type=str, default='../data/data_preprocessed.csv',
                        help='Path to output file (csv)')
    args = parser.parse_args()

    trnf = preprocessing.Compose([
        preprocessing.ToCylindrical(drop_old=True),
        preprocessing.MinMaxScale(columns=['r', 'phi', 'z']),
        preprocessing.DropSpinningTracks(),
        preprocessing.DropFakes()
    ])

    preprocess(path=args.input_file, trnf=trnf, output_path=args.output_file)