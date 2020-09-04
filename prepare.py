import logging
import gin
import pandas as pd

from absl import flags
from absl import app

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
def preprocessor(
        target_model: str,
        output_dir: str
):
    LOGGER.info("GET:", target_model, output_dir)

    # TODO: ugly import hack, remove this import
    from ariadne.RDGraphNet_v1.processor import RDGraphNet_v1_Processor
    df_new = pd.DataFrame()
    a = RDGraphNet_v1_Processor()
    pass


def main(argv):
    del argv
    if FLAGS.config is None:
        raise SystemError("Expected valid path to the GIN-config file supplied as 'config=' parameter")
    gin.parse_config(open(FLAGS.config))
    LOGGER.setLevel(FLAGS.log)
    preprocessor()


if __name__ == '__main__':
    app.run(main)
