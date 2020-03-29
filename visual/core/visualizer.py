import numpy as np
import pandas as pd
from PyQt5.QtCore import QObject, pyqtProperty

from pathlib import Path

from PyQt5.QtQuick import QQuickView
from PyQt5.QtWidgets import QMainWindow
from vispy import scene

from canvas import MyCanvas

TOP_PANEL_OBJ_NAME = "TOP_PANEL_OBJ"

class Visualizer(QObject):
	def __init__(self, cfg, parent):
		super(Visualizer, self).__init__(parent)
		assert cfg["mode"] == "3d"
		self.root_win = None
		self.vispy_canvas = None
		self._current_data_file = None
		self.pd_data = None
		self.top_panel = None
		pass

	def setup_from_vispy_pyqt5(self, root_window: QMainWindow, vispy_canvas: MyCanvas, quick_view: QQuickView):
		self.root_win = root_window
		self.vispy_canvas = vispy_canvas
		self.top_panel = quick_view.rootContext().setContextProperty("visualizer_o", self)

	@pyqtProperty('QString')
	def current_data_file(self):
		return self._current_data_file

	@current_data_file.setter
	def current_data_file(self, value):
		if self._current_data_file != value:
			if not self.open_data_file(value):
				self.current_data_file = self._current_data_file
		self._current_data_file = value

	def open_data_file(self, path: str) -> bool:
		print("OPEN_DATA_FILE")
		my_file = Path(path)
		if not my_file.is_file():
			return False
		dt = np.dtype([('event_id', np.uint32), ('track_id', np.uint32), ('x', float), ('y', float), ('z', float)])
		hits = np.fromfile(path, dtype=dt)
		print("Before constructing dataframe...")
		df = pd.DataFrame(hits)
		# df = df.rename(columns={'track_id':'track'})
		event1 = df[df.event_id == 1]
		event1 = event1.rename(columns={'track_id': 'track'})
		self.pd_data = event1
		print(self.pd_data)
		scatter = scene.Markers()
		scatter.set_data(self.pd_data[['x', 'y', 'z']].values, edge_color=None, face_color=(1, 1, 1, .5), size=5)
		self.vispy_canvas.view.add(scatter)
		return True

