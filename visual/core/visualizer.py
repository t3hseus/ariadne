import numpy as np
import pandas as pd
from PyQt5.QtCore import QObject, pyqtProperty, pyqtSignal, pyqtSlot

from pathlib import Path

from PyQt5.QtQuick import QQuickView
from PyQt5.QtWidgets import QMainWindow
from vispy import scene

from canvas import MyCanvas

TOP_PANEL_OBJ_NAME = "TOP_PANEL_OBJ"

class Visualizer(QObject):

	event_range_changed = pyqtSignal()
	current_data_file_changed = pyqtSignal()
	columns_changed = pyqtSignal()
	cameras_changed = pyqtSignal()

	def __init__(self, cfg, parent):
		super(Visualizer, self).__init__(parent)
		assert cfg["mode"] == "3d"
		self.root_win = None
		self.vispy_canvas = None
		self._current_data_file = None
		self._current_event = -1
		self._event_range = [-1, -1]
		self._columns = []
		self._cameras = []
		self.pd_data = None
		self.top_panel = None
		pass

	def setup_from_vispy_pyqt5(self, root_window: QMainWindow, vispy_canvas: MyCanvas, quick_view: QQuickView):
		self.root_win = root_window
		self.vispy_canvas = vispy_canvas
		self.top_panel = quick_view.rootContext().setContextProperty("visualizer_o", self)

	@pyqtProperty('QString', notify=current_data_file_changed)
	def current_data_file(self):
		return self._current_data_file

	@current_data_file.setter
	def current_data_file(self, value):
		if self._current_data_file != value:
			if self.open_data_file(value):
				self._current_data_file = value
				self.current_data_file_changed.emit()

	@pyqtProperty(int)
	def current_event(self):
		return self._current_event

	@current_event.setter
	def current_event(self, value):
		if self._current_event != value and type(value) is int:
			single_event_data = self.pd_data[self.pd_data.event_id == value]
			data = single_event_data[['x', 'z', 'y']].values
			self.vispy_canvas.add_event_xyz(data, value)
			self._current_event = value

	@pyqtProperty(list, notify=event_range_changed)
	def event_range(self):
		return self._event_range

	@pyqtProperty(list, notify=columns_changed)
	def columns(self):
		return self._columns

	@pyqtProperty(list, notify=cameras_changed)
	def cameras(self):
		return self._cameras

	@pyqtSlot(str, name="set_camera")
	def set_camera(self, camera_name: str):
		if camera_name in MyCanvas.AVAILABLE_CAMERAS:
			self.vispy_canvas.set_camera(camera_name)

	def open_data_file(self, path: str) -> bool:
		print("OPEN_DATA_FILE")
		my_file = Path(path)
		if not my_file.is_file():
			return False
		dt = np.dtype([('event_id', np.uint32), ('track_id', np.uint32), ('x', float), ('y', float), ('z', float)])
		hits = np.fromfile(path, dtype=dt)
		print("Before constructing dataframe...")
		self.pd_data = pd.DataFrame(hits)
		self._event_range = [int(self.pd_data.event_id.min()), int(self.pd_data.event_id.max())]
		self.event_range_changed.emit()
		self._columns = self.pd_data.columns.tolist()
		self.columns_changed.emit()
		self._cameras = MyCanvas.AVAILABLE_CAMERAS
		self.cameras_changed.emit()
		return True

