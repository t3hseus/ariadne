import os

from glob import glob
from importlib import import_module
from argparse import ArgumentParser
from typing import Optional
from data_utils import (
    get_new_fname,
    create_new_folder
)


def root2tsv(path: str, 
             dir_to_save: Optional[str] = None, 
             encoding: str = 'utf-8', 
             sep: str = '\t') -> None:
    """Takes as input path argument to the root file

    If it exists the function extracts data into the dataframe and 
    saves this dataframe in the same location with 
    the same name, but different extension.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"`{path}` is not a file or doesn't exist`")

    result = root2pandas(path)
    fname = get_new_fname(path, dir_to_save)
    print(f"Save data to the `{fname}``")

    if type(result) == tuple:
        fname_params = get_new_fname(path, dir_to_save, path_postfix='params')
        result[0].to_csv(fname, encoding=encoding, sep=sep)
        result[1].to_csv(fname_params, encoding=encoding, sep=sep)
    else:
        result.to_csv(fname, encoding=encoding, sep=sep)

    print("Complete!")


def main(path: str, save_dir: str, encoding: str = 'utf-8', sep: str = '\t') -> None:
    '''Takes as input path argument

    If it exists the function creates a new directory in the
    same location with different name and processes all
    contents of the input directory to save them into 
    a new recently created folder.

    # Arguments
        path: path to the file or directory with files
    '''
    if not os.path.isdir(path):
        raise FileNotFoundError(f"`{path}` is not a directory or doesn't exist`")

    # if directory create another dir in the same location
    print("Creating another folder in the same path")
    dir_to_save = create_new_folder(path, save_dir)

    #process each file and save to dir
    for fpath in glob(os.path.join(path, '*.root')):
        root2tsv(fpath, dir_to_save)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-e', '--experiment', type=str, choices=['bmn'],
        help='Option which defines an experiment')
    parser.add_argument('-p', '--path', type=str, 
        help='Path to the directory with the root files')
    parser.add_argument('--save-dir', type=str, default='tsv',
        help='Postfix that will added to the original dirname to create a new one')
    args = parser.parse_args()
    # import specified function
    module = import_module(args.experiment)
    root2pandas = getattr(module, 'root2pandas')
    # call the main
    main(args.path, args.save_dir)