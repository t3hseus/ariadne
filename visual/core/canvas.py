from collections import namedtuple

import vispy
from vispy.color import Color
from vispy.scene import Node

from vispy.scene.visuals import Polygon, Ellipse, Rectangle, RegularPolygon
from vispy import app, scene
from vispy.app import use_app
from vispy.visuals.shaders import Function
from vispy.visuals.collections import PointCollection

from typing import NamedTuple


class CurrentEvent(NamedTuple):
	event_id: int
	node_data: Node = None

use_app('PyQt5')

class MyCanvas(vispy.scene.SceneCanvas):
	AVAILABLE_CAMERAS = ['turntable', 'arcball', 'panzoom', 'perspective', 'fly']

	def __init__(self, size: (800, 500), watch_dir: str = "."):
		vispy.scene.SceneCanvas.__init__(self, keys='interactive', size=size, bgcolor=Color('#F5F5F5'))
		self.unfreeze()
		self.view = self.central_widget.add_view()
		self.view.camera = 'turntable'  # or try 'arcball'
		self._current_event = CurrentEvent(-1, Node())
		self.freeze()

	def build_axes(self):
		self.unfreeze()

		# add a colored 3D axis for orientation
		ax = scene.Axis(pos=[[0, 0], [1, 0]], tick_direction=(0, -1), font_size=16, parent=self.view.scene,
						axis_color='black', tick_color='black', text_color='black')
		yax = scene.Axis(pos=[[0, 0], [0, 1]], tick_direction=(-1, 0), font_size=16, parent=self.view.scene,
						 axis_color='black', tick_color='black', text_color='black')

		zax = scene.Axis(pos=[[0, 0], [-1, 0]], tick_direction=(0, -1), font_size=16, parent=self.view.scene,
						 axis_color='black', tick_color='black', text_color='black')
		zax.transform = scene.transforms.MatrixTransform()  # its acutally an inverted xaxis
		zax.transform.rotate(90, (0, 1, 0))  # rotate cw around yaxis
		zax.transform.rotate(-45, (0, 0, 1))  # tick direction towards (-1,-1)

		self.freeze()

	def _remove_event(self, event_node: Node):
		event_node.parent = None

	def add_event_xyz(self, data_xyz, event_id):
		if event_id == self._current_event.event_id:
			return
		if self._current_event.node_data is not None:
			self._remove_event(self._current_event.node_data)

		scatter = scene.Markers()
		scatter.set_data(data_xyz, edge_color=(0.0, 0.0, 0.0, .5), face_color=(1, 1, 1, 0.0), size=1)
		self.view.add(scatter)

		self._current_event = CurrentEvent(event_id, scatter)

	def set_camera(self, camera_name: str):
		self.view.camera = camera_name

	@staticmethod
	def run_app():
		vispy.app.run()
