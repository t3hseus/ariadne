import logging
import gin
import pandas as pd

from absl import flags
from absl import app

from ariadne.preprocessing import DataProcessor

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


@gin.configurable
def preprocess(
        target_processor: DataProcessor.__class__,
        output_dir: str
):
    data_df = pd.DataFrame()
    a = target_processor(data_df=data_df)
    LOGGER.info("GET: %r %r" %( a, output_dir ))
    pass


def main(argv):
    del argv
    if FLAGS.config is None:
        raise SystemError("Expected valid path to the GIN-config file supplied as 'config=' parameter")
    gin.parse_config(open(FLAGS.config))
    LOGGER.setLevel(FLAGS.log)
    preprocess()


if __name__ == '__main__':
    app.run(main)
