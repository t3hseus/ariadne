import glob
import logging
import os
from typing import Dict, Tuple, Union

import gin
import numpy as np
import pandas as pd

import ariadne_v2.jit_cacher
from ariadne_v2.jit_cacher import cache_result_df

LOGGER = logging.getLogger('ariadne.parsing')


@cache_result_df()
def parse_df(file_path: str,
             **read_csv_kw_params) -> Tuple[pd.DataFrame, Union[None, str]]:
    return pd.read_csv(file_path, **read_csv_kw_params)


def parse_single_arr_arg(arr_arg):
    if '..' in arr_arg:
        args = arr_arg.split('..')
        assert len(args) == 2, "It should have form '%num%..%num%' ."
        return np.arange(int(args[0]), int(args[1])), False
    if ':' in arr_arg:
        return -1, True
    return [int(arr_arg)], False


@gin.configurable
def parse(input_file_mask,
          csv_params: Dict[str, object],
          events_quantity,
          filter_func=None):
    files_list = glob.glob(input_file_mask)
    assert len(files_list) > 0, f"no files found matching mask {input_file_mask}"
    assert isinstance(events_quantity, str), 'events_quantity should be a str. see comments in config to set it ' \
                                             'correctly. Got: %r with type %r ' % (
                                                 events_quantity, type(events_quantity))
    event_idxs, parse_all = parse_single_arr_arg(events_quantity)
    list = '\n'.join(files_list)
    LOGGER.info(f"[Parse]: matched {len(files_list)} files: {list}\n")
    for idx, elem in enumerate(files_list):
        LOGGER.info("[Parse]: started parsing CSV #%d (%s):" % (idx, elem))
        parsed_df, hash = parse_df(elem, **csv_params)
        hash = ariadne_v2.jit_cacher.Cacher.build_hash(events_quantity, hash)
        if filter_func:
            parsed_df = filter_func(parsed_df)
            hash = None
        LOGGER.info("[Parse]: finished parsing CSV...")
        if not parse_all:
            res = np.array(event_idxs)
            yield parsed_df[parsed_df.event.isin(res)].copy(), os.path.basename(elem), hash
            return
        else:
            yield parsed_df, os.path.basename(elem), hash
    return
