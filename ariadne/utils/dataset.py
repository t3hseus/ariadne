import glob
import numpy as np

def load_data(input_dir, file_mask, n_samples=None):
    '''Function to load input data for tracknet-like models training.
    Helps load prepared input of tracknet or classiier (like inputs,
    input_lengths, labels etc) or last station hits stored somewhere
    as npz-file (with info about positions and events).

    Arguments:
        input_dir (string): directory with files
        file_mask (string): mask to be applied to (f.e. *.npz)
        n_samples (int, None by default): Maximum number of samples to be loaded
    '''
    flag = 0
    files = []
    data_merged = {}
    datafiles = glob.glob(f'{input_dir}/{file_mask}')
    for f in datafiles:
        one_file_data = np.load(f, mmap_mode='r', allow_pickle=True)
        first_file_len = len(one_file_data[one_file_data.files[0]])
        files = one_file_data.files
        for f in files:
            print(one_file_data[f][0])
        lengths = [len(one_file_data[f]) == first_file_len for f in files]
        assert all(lengths), 'Lengths of files in npz are not equal!'
        if flag == 0:
            data_merged = dict(one_file_data.items())
            flag = 1
        else:
            for k, i in one_file_data.items():
                data_merged[k] = np.concatenate((data_merged[k], i), 0)
    if n_samples is None:
        n_samples = len(data_merged[files[0]])
    else:
        assert isinstance(n_samples, int), 'n_samples must be int or None'
        n_samples = min(n_samples, len(data_merged[files[0]]))
    return {key: item[:n_samples] for key, item in data_merged.items()}