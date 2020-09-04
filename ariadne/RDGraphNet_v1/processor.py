import logging

from ariadne.preprocessing import EventProcessor

import gin
import pandas as pd

LOGGER = logging.getLogger('ariadne.prepare')


@gin.configurable
class RDGraphNet_v1_Processor(EventProcessor):

    def __init__(self,
                 processor_name: str = "RDGraphNet_v1_Processor",
                 some_custom_field: str = "some_field",
                 data_df: pd.DataFrame = pd.DataFrame(),
                 ):
        super().__init__(processor_name, data_df)
        LOGGER.info("GOT %s \n%s \n%s\n========", data_df, processor_name, some_custom_field)
