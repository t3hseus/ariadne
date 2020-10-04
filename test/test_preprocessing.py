import os
import unittest
import sys
from collections import Iterable
from os import listdir
from os.path import isfile, join

import gin
import pandas as pd
import numpy as np

from ariadne.graph_net.graph_utils.graph import load_graph
from ariadne.graph_net.processor import GraphNet_Processor
from ariadne.parsing import parse_df
from prepare import parse, preprocess

class GinconfigSaver(object):
    def __init__(self, params):
        self.old_vals = params
        for key in self.old_vals.keys():
            self.old_vals[key] = gin.query_parameter(key)
        pass

    def __del__(self):
        for key, value in self.old_vals.items():
            gin.bind_parameter(key, value)

class StandardTestCase(unittest.TestCase):
    CONFIG_PATH = 'test/gin/rdgraphnet_prepare_test.cfg'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        gin.parse_config(open(self.CONFIG_PATH))
        self.graphs_path = ''
        self.cfg_path = ''

    @gin.configurable(module='StandardTestCase')
    def get_processor(self,
                      target_processor,
                      output_dir,
                      input_file,
                      csv_params,
                      events_quantity,
                      constructed_graphs_path,
                      prepare_cfg_path
                      ):
        df = parse()
        df_test_data = parse(input_file=input_file,
                             csv_params=csv_params,
                             events_quantity=events_quantity)
        self.assertTrue(df_test_data.r.values.max() < 1.)
        self.assertEqual(df.event.nunique(), 11)
        processor = target_processor(data_df=df,
                                     output_dir=output_dir)
        self.assertTrue(os.path.exists(constructed_graphs_path))
        self.graphs_path = constructed_graphs_path
        self.cfg_path = prepare_cfg_path
        return processor, df_test_data

    @staticmethod
    def get_graphs(graph_folder):
        graphs = {}
        graphs_files = [f for f in listdir(graph_folder)
                        if isfile(join(graph_folder, f)) and 'graph_' in f]
        for graph_file in graphs_files:
            graphs[graph_file] = load_graph(join(graph_folder, graph_file))

        return graphs

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

    def test_GraphNet_Processor_transformations_values(self):
        processor, df_test_data = self.get_processor()
        processor_transformer = processor.transformer
        data_df = processor.data_df

        target_events = data_df  # [data_df.event == 0]
        source_events = df_test_data  # [df_test_data.event == 0]
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

    def test_GraphNet_Processor_created_graph_values(self):

        processor, df_test_data = self.get_processor()
        saver = GinconfigSaver(
            {
                'parse.events_quantity': object(),
                'parse.csv_params': object(),
                'parse.input_file': object(),
                'StandardTestCase.get_processor.output_dir': object()
            }
        )

        source_loaded_graphs = self.get_graphs(self.graphs_path)
        self.assertTrue(len(source_loaded_graphs) == 50)
        gin.parse_config_file(self.cfg_path)
        gin.bind_parameter('parse.events_quantity', ['0..50'])
        gin.bind_parameter('parse.csv_params', {
            "sep": '\s+',
            "encoding": 'utf-8',
            "names": ['event', 'x', 'y', 'z', 'station', 'track']
        })
        gin.bind_parameter('parse.input_file', 'resources/test_data/cgem_50_events.txt')

        gin.bind_parameter('StandardTestCase.get_processor.output_dir', 'output/tests/cgem_graphnet_test')
        gin.bind_parameter('preprocess.output_dir', 'output/tests/cgem_graphnet_test')
        preprocess()

        target_loaded_graphs = self.get_graphs('output/tests/cgem_graphnet_test')
        self.assertTrue(len(target_loaded_graphs) == 50)

        for graph_name, graph_object in source_loaded_graphs.items():
            self.assertTrue(graph_name in target_loaded_graphs)

            graph_obj_source = source_loaded_graphs[graph_name]
            graph_obj_target = target_loaded_graphs[graph_name]

            np.testing.assert_allclose(
                graph_obj_source.X,
                graph_obj_target.X,
                err_msg='on graph %s' % graph_name
            )

            np.testing.assert_allclose(
                graph_obj_source.y,
                graph_obj_target.y,
                err_msg='on graph %s' % graph_name
            )

            np.testing.assert_allclose(
                graph_obj_source.Ri,
                graph_obj_target.Ri,
                err_msg='on graph %s' % graph_name
            )

            np.testing.assert_allclose(
                graph_obj_source.Ro,
                graph_obj_target.Ro,
                err_msg='on graph %s' % graph_name
            )


if __name__ == '__main__':
    unittest.main()
