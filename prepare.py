import logging
import os
from typing import Dict

import gin
import pandas as pd
import numpy as np

from absl import flags
from absl import app

from ariadne.preprocessing import DataProcessor
from parsing import parse_df

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


# todo: made universal parse
@gin.configurable
def parse(
        input_file: str,
        csv_params: Dict[str, object],
        events_quantity
):
    parsed_df = parse_df(input_file,
                    **csv_params)

    def parseSingleArrArg(arrArg):
        if '..' in arrArg:
            args = arrArg.split('..')
            assert len(args) == 2 , "It should have form '%num%..%num%' ."
            return np.arange(int(args[0]), int(args[1])), False
        if ':' in arrArg:
            return -1, True
        return [int(arrArg)], False

    res = np.array([])
    for elem in events_quantity:
        toAppend, is_all = parseSingleArrArg(elem)
        if is_all:
            return parsed_df
        res = np.append(res, toAppend)

    return parsed_df[parsed_df.event.isin(res)].copy()
#endof TODO

@gin.configurable
def preprocess(
        target_processor: DataProcessor.__class__,
        output_dir: str
):
    os.makedirs(output_dir, exist_ok=True)

    data_df = parse()
    processor: DataProcessor = target_processor(data_df=data_df,
                                                output_dir=output_dir)

    generator = processor.generate_chunks_iterable()

    preprocessed_chunks = []

    for (idx, df_chunk) in generator:
        data_chunk = processor.construct_chunk(df_chunk)
        preprocessed_chunks.append(
            processor.preprocess_chunk(chunk=data_chunk, idx=idx)
        )

    processed_data = processor.postprocess_chunks(preprocessed_chunks)
    processor.save_on_disk(processed_data)


def main(argv):
    del argv
    if FLAGS.config is None:
        raise SystemError("Expected valid path to the GIN-config file supplied as 'config=' parameter")
    gin.parse_config(open(FLAGS.config))
    LOGGER.setLevel(FLAGS.log)
    preprocess()


if __name__ == '__main__':
    app.run(main)
