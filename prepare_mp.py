import copy
import logging
import multiprocessing
import os
import os.path
import traceback

from ariadne_v2 import jit_cacher
from ariadne_v2.dataset import AriadneDataset
from ariadne_v2.inference import IPreprocessor, Transformer, IPostprocessor
from ariadne_v2.jit_cacher import Cacher
from ariadne_v2.preprocessing import DFDataChunk

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


class EventProcessor(multiprocessing.Process):
    def __init__(self,
                 df_hash,
                 basename,
                 target_processor,
                 target_postprocessor,
                 main_dataset,
                 work_slice,
                 idx,
                 result_queue: multiprocessing.Queue,
                 message_queue: multiprocessing.Queue,
                 global_lock: multiprocessing.Lock(),
                 **kwargs):
        super(EventProcessor, self).__init__(**kwargs)
        self.idx = idx
        self.basename = basename
        self.target_processor = target_processor
        self.target_postprocessor = target_postprocessor
        self.main_dataset = main_dataset
        self.df_hash = df_hash
        self.work_slice = work_slice
        self.result_queue = result_queue
        self.message_queue = message_queue
        self.global_lock = global_lock

    def run(self):
        jit_cacher.init_locks(self.global_lock)

        with jit_cacher.instance() as cacher:
            cacher.init()
            data_df = cacher.read_df(self.df_hash)
            if data_df.empty:
                self.result_queue.put([self.idx])
                return

        old_path = self.main_dataset.dataset_path
        with self.main_dataset.create(cacher, old_path + f"_p{self.idx}", ) as process_dataset:
            data_df = data_df[(data_df.event >= self.work_slice[0]) & (data_df.event < self.work_slice[1])]
            for ev_id, event in data_df.groupby('event'):
                try:
                    chunk = DFDataChunk.from_df(event)
                    processed = self.target_processor(chunk)
                    if processed is None:
                        return None, ev_id
                    idx = f"graph_{self.basename}_{ev_id}"
                    postprocessed = self.target_postprocessor(processed, process_dataset, idx)
                    self.result_queue.put([-1])
                except:
                    stack = traceback.format_exc()
                    print(f"Exception in process {os.getpid()}! details below: {stack}")
                    self.result_queue.put([-2, stack])
                    break
                if not self.message_queue.empty:
                    break

            process_dataset.dataset_path = old_path
            process_dataset.submit(f"p_{self.idx}")
        self.result_queue.put([self.idx])
        #
        # ev_id, event = data
        # try:
        #     chunk = DFDataChunk.from_df(event)
        #     processed = self.target_processor(chunk)
        #     if processed is None:
        #         return None, ev_id
        #
        #     idx = f"graph_{self.basename}_{ev_id}"
        #     # postprocessed = self.target_postprocessor(processed, self.target_dataset, idx)
        #     return True, ev_id
        # except Exception as exc:
        #     stack = traceback.format_exc()
        #     LOGGER.info(f"Exception in process {os.getpid()}! details below: {stack}")
        #     return False, ev_id, exc, stack

    # def submit(self):
    #     self.target_dataset.submit()

    # def __del__(self):
    #     print(f"ON FINI!!! my poc{os.getpid()}")


@gin.configurable
def preprocess_mp(
        transformer: Transformer,
        target_processor: IPreprocessor,
        target_postprocessor: IPostprocessor,
        target_dataset: AriadneDataset,
        process_num: int = None,
        chunk_size: int = 1
):
    os.makedirs(target_dataset.dataset_path, exist_ok=True)
    setup_logger(target_dataset.dataset_path, target_processor.__class__.__name__)

    global_lock = multiprocessing.Lock()
    jit_cacher.init_locks(global_lock)

    if True:
        with jit_cacher.instance() as cacher:
            cacher.init()

        with target_dataset.create(cacher) as ds:
            target_dataset = ds

    target_dataset.meta["cfg"] = gin.config_str()

    # warnings to exceptions:
    pd.set_option('mode.chained_assignment', 'raise')

    LOGGER.info("GOT config: \n======config======\n %s \n========config=======" % gin.config_str())
    process_num = multiprocessing.cpu_count() if process_num is None else process_num
    LOGGER.info(f"Running with the {process_num} processes with chunk_size={chunk_size}")

    for data_df, basename, df_hash in parse():
        LOGGER.info("[Preprocess]: started processing a df with %d rows:" % len(data_df))

        data_df, hash = transformer(DFDataChunk.from_df(data_df, df_hash), return_hash=True)
        event_count = data_df.event.nunique()
        chunk_size = event_count // process_num
        if event_count // process_num == 0:
            process_num = 1
            chunk_size = 1

        result_queue = multiprocessing.Queue()
        message_queue = multiprocessing.Queue()
        workers = []
        for i in range(0, process_num):
            work_slice = (i * chunk_size, (i + 1) * chunk_size)
            workers.append(EventProcessor(hash,
                                          basename,
                                          target_processor,
                                          target_postprocessor,
                                          target_dataset, work_slice, i, result_queue, message_queue, global_lock))
            workers[-1].start()

        try:
            with tqdm(total=int(data_df.event.nunique())) as pbar:
                while any(workers) > 0:
                    obj = result_queue.get()
                    if obj[0] == -1:
                        pbar.update()
                    elif obj[0] == -2:
                        LOGGER.info(f"Process got exception: {obj[1]}.")
                        return
                    else:
                        LOGGER.info(f"Process idx={obj} has finished processing. joining...")
                        workers[obj].join()
                        workers[obj].close()
                        workers[obj] = False
        except KeyboardInterrupt:
            LOGGER.info("KeyboardInterrupt! terminating all processes....")
            message_queue.put(1)
            [worker.join() for worker in workers if worker]

        # with tqdm(total=int(data_df.event.nunique())) as pbar:
        #     for ret in pool.imap_unordered(processor, data_df.groupby('event'), chunksize=chunk_size):
        #         pbar.update()
        #         if ret[0] == False:
        #             LOGGER.info(f"[Preprocess]: exception in event {ret[1]}.\n"
        #                         f"Exc: {ret[2]}\n"
        #                         f"Stack: {ret[3]}")
        #             return
        #         if ret[0] is None:
        #             LOGGER.info(f"[Preprocess]: bad event {ret[1]}")


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
