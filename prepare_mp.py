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
                 global_lock: multiprocessing.Lock,
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
        try:
            jit_cacher.init_locks(self.global_lock)

            with jit_cacher.instance() as cacher:
                cacher.init()
                data_df = cacher.read_df(self.df_hash)
                if data_df.empty:
                    self.result_queue.put([self.idx, ''])
                    return

            old_path = self.main_dataset.dataset_name
            new_path = old_path + f"_{self.basename}_p{self.idx}"
        except KeyboardInterrupt:
            print(f"KeyboardInterrupt in process {os.getpid()}. No result will be returned.")
            self.result_queue.put([self.idx, ''])
            jit_cacher.fini_locks()
            return

        try:
            with self.main_dataset.open_dataset(cacher, new_path) as process_dataset:
                data_df = data_df[(data_df.event >= self.work_slice[0]) & (data_df.event < self.work_slice[1])]
                # print(f"DATA_DF: {data_df.event.nunique()} pid:{os.getpid()}, slice: {self.work_slice}")
                for ev_id, event in data_df.groupby('event'):
                    try:
                        chunk = DFDataChunk.from_df(event)
                        processed = self.target_processor(chunk)
                        if processed is None:
                            continue
                        idx = f"graph_{self.basename}_{ev_id}"
                        postprocessed = self.target_postprocessor(processed, process_dataset, idx)
                        self.result_queue.put([-1])
                    except KeyboardInterrupt:
                        raise KeyboardInterrupt
                    except:
                        stack = traceback.format_exc()
                        print(f"Exception in process {os.getpid()}! details below: {stack}")
                        self.result_queue.put([-2, stack])
                        break
                    if not self.message_queue.empty:
                        break
        except KeyboardInterrupt:
            print(f"KeyboardInterrupt in process {os.getpid()}.")

        process_dataset.dataset_name = old_path
        process_dataset.local_submit()

        # finish signal
        self.result_queue.put([self.idx, new_path])

        jit_cacher.fini_locks()


@gin.configurable
def preprocess_mp(
        transformer: Transformer,
        target_processor: IPreprocessor,
        target_postprocessor: IPostprocessor,
        target_dataset: AriadneDataset,
        process_num: int = None,
        chunk_size: int = 1
):
    os.makedirs(target_dataset.dataset_name, exist_ok=True)
    setup_logger(target_dataset.dataset_name, target_processor.__class__.__name__)

    global_lock = multiprocessing.Lock()
    jit_cacher.init_locks(global_lock)

    if True:
        with jit_cacher.instance() as cacher:
            cacher.init()

        with target_dataset.open_dataset(cacher) as ds:
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
        events = sorted(list(data_df.event.unique()))
        chunk_size = event_count // process_num
        if event_count // process_num == 0:
            process_num = 1
            chunk_size = 1

        result_queue = multiprocessing.Queue()
        message_queue = multiprocessing.Queue()
        workers = []
        workers_result = [""] * process_num
        for i in range(0, process_num):

            if i == process_num - 1:
                work_slice = (events[i * chunk_size], 1e10)
            else:
                work_slice = (events[i * chunk_size], events[(i + 1) * chunk_size])

            workers.append(EventProcessor(hash,
                                          basename,
                                          target_processor,
                                          target_postprocessor,
                                          target_dataset, work_slice, i, result_queue, message_queue, global_lock))
            workers[-1].start()
        canceled = False
        try:
            pbar = tqdm(total=len(events))
            while any(workers):
                obj = result_queue.get()
                if obj[0] == -1:
                    pbar.update()
                elif obj[0] == -2:
                    LOGGER.info(f"Process got exception: {obj[1]}.")
                    return
                else:
                    pbar.update()
                    LOGGER.debug(f"Process idx={obj} has finished processing. joining...")
                    workers[obj[0]].join()
                    workers[obj[0]].close()
                    workers[obj[0]] = False
                    workers_result[obj[0]] = obj[1]
        except KeyboardInterrupt:
            LOGGER.info("KeyboardInterrupt! terminating all processes....")
            message_queue.put(1)
            [worker.join() for worker in workers if worker]
            canceled = True

        LOGGER.info("Finished processing. Merging results....")

        while not result_queue.empty():
            obj = result_queue.get()
            if obj[0] >= 0:
                workers_result[obj[0]] = obj[1]

        for worker_id, worker_result in enumerate(workers_result):
            if worker_result == "":
                LOGGER.info(f"Worker {worker_id} failed...")

        workers_result = [worker_result for worker_result in workers_result if worker_result != '']
        with target_dataset.open_dataset(cacher, target_dataset.dataset_name, drop_old=False) as ds:
            ds.global_submit(workers_result)

        if canceled:
            break


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
