import glob
import logging
import os
import os.path
from typing import Dict

os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

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
def parse(
        input_file_mask,
        csv_params: Dict[str, object],
        events_quantity
):
    files_list = glob.glob(input_file_mask)
    assert len(files_list) > 0, f"no files found matching mask {input_file_mask}"
    assert isinstance(events_quantity, str), 'events_quantity should be a str. see comments in config to set it ' \
                                               'correctly. Got: %r with type %r ' % (events_quantity, type(events_quantity))
    event_idxs, parse_all = parse_single_arr_arg(events_quantity)
    print(event_idxs)
    LOGGER.info("[Parse]: matched following files:")
    LOGGER.info("[Parse]: %r" % files_list)
    for idx, elem in enumerate(files_list):
        LOGGER.info("[Parse]: started parsing CSV #%d (%s):" % (idx, elem))
        parsed_df = parse_df(elem,**csv_params)
        LOGGER.info("[Parse]: finished parsing CSV...")
        if not parse_all:
            res = np.array(event_idxs)
            yield parsed_df[parsed_df.event.isin(res)].copy(), os.path.basename(elem)
            return
        else:
            yield parsed_df, os.path.basename(elem)
    return
# endof TODO

@gin.configurable
def preprocess(
        target_processor: DataProcessor.__class__,
        output_dir: str,
        ignore_asserts: False
):
    os.makedirs(output_dir, exist_ok=True)

    for data_df, basename in parse():
        LOGGER.info("[Preprocess]: started processing a df with %d rows:" % len(data_df))
        processor: DataProcessor = target_processor(data_df=data_df,
                                                    output_dir=output_dir)

        generator = processor.generate_chunks_iterable()

        preprocessed_chunks = []
        try:
            for (idx, df_chunk) in tqdm(generator):
                try:
                    data_chunk = processor.construct_chunk(df_chunk)
                except AssertionError as ex:
                    if ignore_asserts:
                        LOGGER.warning("GOT ASSERT %r on idx %d" % (ex, idx))
                        continue
                    else:
                        raise ex
                preprocessed_chunks.append(
                    processor.preprocess_chunk(chunk=data_chunk, idx=basename)
                )
        except KeyboardInterrupt as ex:
            LOGGER.warning("BREAKING by interrupt. got %d processed chunks" % len(preprocessed_chunks))
        processed_data = processor.postprocess_chunks(preprocessed_chunks)
        processor.save_on_disk(processed_data)


def main(argv):
    del argv
    if FLAGS.config is None:
        raise SystemError("Expected valid path to the GIN-config file supplied as '--config %PATH%' parameter")
    gin.parse_config(open(FLAGS.config))
    LOGGER.setLevel(FLAGS.log)
    LOGGER.info("GOT config: \n======config======\n %s \n========config=======" % gin.config_str())
    preprocess()


if __name__ == '__main__':
    app.run(main)
