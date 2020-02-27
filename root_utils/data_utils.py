import shutil
import os

from datetime import datetime
from typing import Optional


def get_new_fname(fname: str, 
                  path_prefix: Optional[str]=None, 
                  path_postfix: Optional[str]=None) -> str:
    '''Creates a new input fname using the original
    replacing the extension to tsv

    # Arguments
        fname: original file name or path to file
        path_prefix: pwd path for the new file
        path_postfix: filename postfix

    # Returns
        new file's name
    '''
    fpath, fname = os.path.split(fname)
    fname, ext = os.path.splitext(fname)
    
    if path_prefix is None:
        fname = os.path.join(fpath, fname)
    else:
        fname = os.path.join(path_prefix, fname)

    if path_postfix is not None:
        fname = '_'.join([fname, path_postfix])

    fname = '.'.join([fname, 'tsv'])
    return fname


def create_new_folder(path, path_postfix, rmdir_old=True):
    '''Creates a new folder using the input path by adding
    to it a `path_postfix`. 

    If the directory with the choosed name exists 
    removes it or not depending on `rmdir_old` parameter

    # Arguments
        path: string, path to the original directory
        path_postfix: string, will be added to original to create new name
        rmdir_old: boolean, whether or not to remove old dir

    # Returns 
        string, name of the created folder
    '''
    dir_to_save = '_'.join([path, path_postfix])

    if not os.path.isdir(dir_to_save):
        os.mkdir(dir_to_save)
        print("Folder '%s' was created" % dir_to_save)
        return dir_to_save

    if rmdir_old:
        print("Folder with such name exists, remove it")
        shutil.rmtree(dir_to_save)

    if not rmdir_old:
        datetime_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dir_to_save = '_'.join([dir_to_save, datetime_now])

    os.mkdir(dir_to_save)
    print("Folder '%s' was created" % dir_to_save)
    return dir_to_save