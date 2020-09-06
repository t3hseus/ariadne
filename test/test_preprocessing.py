import unittest
import sys
from collections import Iterable

import gin
import pandas as pd
import numpy as np

from ariadne.graph_net.processor import GraphNet_Processor
from ariadne.parsing import parse_df
from prepare import parse


class StandardTestCase(unittest.TestCase):
    CONFIG_PATH = 'gin/rdgraphnet_prepare_test.cfg'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        gin.parse_config(open(self.CONFIG_PATH))

    @gin.configurable(module='StandardTestCase')
    def get_processor(self,
                      target_processor,
                      output_dir,
                      input_file,
                      csv_params,
                      events_quantity
                      ):
        df = parse()
        df_test_data = parse(input_file=input_file,
                             csv_params=csv_params,
                             events_quantity=events_quantity)
        self.assertTrue(df_test_data.r.values.max() < 1.)
        self.assertEqual(df.event.nunique(), 11)
        processor = target_processor(data_df=df,
                                     output_dir=output_dir)
        return processor, df_test_data

    def test_GraphNet_Processor_iterable(self):
        processor, df_test_data = self.get_processor()
        chunks_iterable = processor.generate_chunks_iterable()
        self.assertIsInstance(chunks_iterable, Iterable)

    def test_GraphNet_Processor_iterable_1(self):
        processor, df_test_data = self.get_processor()
        chunks_iterable = processor.generate_chunks_iterable()
        self.assertIsInstance(chunks_iterable, Iterable)

    def test_GraphNet_Processor_transformations_names(self):
        processor, df_test_data = self.get_processor()
        processor_transformer = processor.transformer
        data_df = processor.data_df

        event_n0 = data_df[data_df.event == 0]
        transformed_event = processor_transformer(event_n0)
        self.assertEqual(set(transformed_event.columns),
                         set(df_test_data.columns))
        # self.assertTrue(np.allclose(transformed_event.columns,
        #                             df_txest_data.columns))

    def test_GraphNet_Processor_transformations_values(self):
        processor, df_test_data = self.get_processor()
        processor_transformer = processor.transformer
        data_df = processor.data_df

        target_events = data_df #[data_df.event == 0]
        source_events = df_test_data #[df_test_data.event == 0]
        transformed_event = processor_transformer(target_events)
        TARGET_COLUMNS = ['r', 'phi', 'z']
        self.assertTrue(set(TARGET_COLUMNS).issubset(set(transformed_event.columns)))
        self.assertTrue(len(target_events) == len(source_events))
        target_df = target_events[TARGET_COLUMNS]
        source_df = source_events[TARGET_COLUMNS]
        for index, row in target_df.iterrows():
            self.assertTrue(np.allclose(
                row.values,
                source_df.iloc[index].values
            ), "not equal at row %d\n with values:\n 0:\t\t%r\n 1:\t\t%r " % (index,
                                                                                       row.values,
                                                                                       source_df.iloc[index].values))

if __name__ == '__main__':
    unittest.main()
