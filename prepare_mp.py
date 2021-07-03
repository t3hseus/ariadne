import logging
import multiprocessing
import os
import os.path

from ariadne_v2.inference import IPreprocessor, Transformer, IPostprocessor
from ariadne_v2.preprocessing import DataChunk

os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import gin
import pandas as pd

from absl import flags
from absl import app
from tqdm import tqdm
from ariadne_v2.parsing import parse

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
    logger_dir = os.path.join(logger_dir, "logs_"+preprocessor_name)
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

class EventProcessor:
    def __init__(self,
                 output_dir,
                 basename,
                 target_processor,
                 target_postprocessor):

        self.output_dir = output_dir
        self.basename = basename
        self.target_processor = target_processor
        self.target_postprocessor = target_postprocessor

    def __call__(self, data):
        ev_id, event = data
        idx = os.path.join(self.output_dir, f"graph_{self.basename}_{ev_id}")
        chunk = DataChunk.from_df(event)
        processed = self.target_processor(chunk)
        if processed is None:
            return False, ev_id
        postprocessed = self.target_postprocessor(processed, idx)
        return True, ev_id



@gin.configurable
def preprocess_mp(
        transformer: Transformer.__class__,
        target_processor: IPreprocessor.__class__,
        target_postprocessor: IPostprocessor.__class__,
        output_dir: str,
        process_num: int = None,
        chunk_size: int = 1
):
    os.makedirs(output_dir, exist_ok=True)
    setup_logger(output_dir, target_processor.__class__.__name__)

    # warnings to exceptions:
    pd.set_option('mode.chained_assignment', 'raise')

    LOGGER.info("GOT config: \n======config======\n %s \n========config=======" % gin.config_str())
    process_num = multiprocessing.cpu_count() if process_num is None else process_num
    LOGGER.info(f"Running with the {process_num} processes with chunk_size={chunk_size}")
    pool = multiprocessing.Pool(processes=process_num)

    for data_df, basename in parse():
        LOGGER.info("[Preprocess]: started processing a df with %d rows:" % len(data_df))
        processor = EventProcessor(
                 output_dir,
                 basename,
                 target_processor,
                 target_postprocessor)
        data_df = transformer(data_df)
        with tqdm(total=int(data_df.event.nunique())) as pbar:
            for ret in pool.imap_unordered(processor, data_df.groupby('event'), chunksize=chunk_size):
                pbar.update()
                if ret[0] is None:
                    LOGGER.info(f"[Preprocess]: bad event {ret[1]}")



def main(argv):
    del argv
    if FLAGS.config is None:
        raise SystemError("Expected valid path to the GIN-config file supplied as '--config %PATH%' parameter")
    gin.parse_config(open(FLAGS.config))
    LOGGER.setLevel(FLAGS.log)
    preprocess_mp()
    LOGGER.info("end processing")

if __name__ == '__main__':
    app.run(main)
