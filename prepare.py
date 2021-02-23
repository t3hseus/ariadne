import glob
import logging
import multiprocessing
import os
import os.path
import sys
import traceback
from functools import partial
from multiprocessing import Pool
from typing import Dict

os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
import _gin_bugfix

import gin
import pandas as pd
import numpy as np

from absl import flags
from absl import app
from tqdm import tqdm

from ariadne.preprocessing import DataProcessor
from ariadne.parsing import parse_df

FLAGS = flags.FLAGS
flags.DEFINE_string(
    name='config', default=None,
    help='Path to the config file to use.'
)
flags.DEFINE_enum(
    name='log', default='INFO',
    enum_values=['INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL'],
    help='Level of logging'
)


def setup_logger(logger_dir, preprocessor_name):
    # create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    logger_dir = os.path.join(logger_dir, "logs_" + preprocessor_name)
    os.makedirs(logger_dir, exist_ok=True)
    fh = logging.FileHandler('%s/prepare_%s.log' % (logger_dir, preprocessor_name))
    fh.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')

    # add formatter to ch
    fh.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(fh)


LOGGER = logging.getLogger('ariadne.prepare')


def parse_single_arr_arg(arr_arg):
    if '..' in arr_arg:
        args = arr_arg.split('..')
        assert len(args) == 2, "It should have form '%num%..%num%' ."
        return np.arange(int(args[0]), int(args[1])), False
    if ':' in arr_arg:
        return -1, True
    return [int(arr_arg)], False


# todo: made universal parse
@gin.configurable
def parse(input_file_mask,
          csv_params: Dict[str, object],
          events_quantity,
          filter_func=None
          ):
    files_list = glob.glob(input_file_mask)
    assert len(files_list) > 0, f"no files found matching mask {input_file_mask}"
    assert isinstance(events_quantity, str), 'events_quantity should be a str. see comments in config to set it ' \
                                             'correctly. Got: %r with type %r ' % (
                                                 events_quantity, type(events_quantity))
    event_idxs, parse_all = parse_single_arr_arg(events_quantity)
    LOGGER.info("[Parse]: matched following files:")
    LOGGER.info("[Parse]: %r" % files_list)
    for idx, elem in enumerate(files_list):
        LOGGER.info("[Parse]: started parsing CSV #%d (%s):" % (idx, elem))
        parsed_df = parse_df(file_path=elem, **csv_params)
        if filter_func:
            parsed_df = filter_func(parsed_df)
        LOGGER.info("[Parse]: finished parsing CSV...")
        if not parse_all:
            res = np.array(event_idxs)
            yield parsed_df[parsed_df.event.isin(res)].copy(), os.path.basename(elem)
            return
        else:
            yield parsed_df, os.path.basename(elem)
    return


# endof TODO

def proc(processor: DataProcessor, ignore_asserts, basename, data):
    idx, chunk_df = data
    try:
        try:
            data_chunk = processor.construct_chunk(chunk_df)
        except AssertionError as ex:
            if ignore_asserts:
                return [None, None, idx, AssertionError("".join(traceback.format_exception(*sys.exc_info())))]
            else:
                raise AssertionError("".join(traceback.format_exception(*sys.exc_info())))
        res = processor.preprocess_chunk(chunk=data_chunk, idx=basename)
    except Exception:
        return [None, None, idx, Exception("".join(traceback.format_exception(*sys.exc_info())))]
    return [res, processor.forward_fields(), idx, None]


@gin.configurable
def preprocess(
        target_processor: DataProcessor.__class__,
        output_dir: str,
        ignore_asserts: bool,
        chunk_length: int,
        cpu_count
):
    os.makedirs(output_dir, exist_ok=True)
    setup_logger(output_dir, target_processor.__name__)

    LOGGER.info("GOT config: \n======config======\n %s \n========config=======" % gin.config_str())

    for data_df, basename in parse():
        LOGGER.info("[Preprocess]: started processing a df with %d rows:" % len(data_df))
        processor: DataProcessor = target_processor(output_dir=output_dir)

        generator = processor.generate_chunks_iterable(data_df)
        total = processor.total_chunks(data_df)

        preprocessed_chunks = []
        forwarded_fields = []

        with Pool(processes=cpu_count if cpu_count > 0 else multiprocessing.cpu_count() - 1) as pool:
            funcc = partial(proc, processor, ignore_asserts, basename)
            with tqdm(total=total) as pbar:
                try:
                    for data, forward_fields, idx, ex in pool.imap_unordered(funcc, generator, chunk_length):
                        pbar.update()
                        if data is not None:
                            preprocessed_chunks.append(data)
                            forwarded_fields.append(forward_fields)
                        else:
                            LOGGER.warning(f"Got exception from processor on idx={idx}. Exception info:")
                            LOGGER.warning(ex)
                            LOGGER.warning(f"End exception info\n")

                except KeyboardInterrupt as ex:
                    pool.terminate()
                    pool.join()
                    LOGGER.warning("BREAKING by interrupt. got %d processed chunks" % len(preprocessed_chunks))

        for fields_bunch in forwarded_fields:
            processor.reduce_fields(fields_bunch)
        processed_data = processor.postprocess_chunks(preprocessed_chunks)
        processor.save_on_disk(processed_data)


def main(argv):
    del argv
    if FLAGS.config is None:
        raise SystemError("Expected valid path to the GIN-config file supplied as '--config %PATH%' parameter")
    gin.parse_config(open(FLAGS.config))
    LOGGER.setLevel(FLAGS.log)
    preprocess()
    LOGGER.info("end processing")


if __name__ == '__main__':
    app.run(main)
