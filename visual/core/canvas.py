import vispy
from vispy.color import Color

from vispy.scene.visuals import Polygon, Ellipse, Rectangle, RegularPolygon
from vispy import app, scene
from vispy.app import use_app
from vispy.visuals.shaders import Function
from vispy.visuals.collections import PointCollection


class MyCanvas(vispy.scene.SceneCanvas):

	def __init__(self, size: (800, 500), watch_dir: str = "."):
		vispy.scene.SceneCanvas.__init__(self, keys='interactive', size=size, bgcolor=Color('#F5F5F5'))
		self.unfreeze()
		self.view = self.central_widget.add_view()
		self.view.camera = 'turntable'  # or try 'arcball'
		self.freeze()

	def build_axes(self):
		self.unfreeze()

		# add a colored 3D axis for orientation
		ax = scene.Axis(pos=[[0, 0], [1, 0]], tick_direction=(0, -1), font_size=16, parent=self.view.scene,
						axis_color='black', tick_color='black', text_color='black')
		yax = scene.Axis(pos=[[0, 0], [0, 1]], tick_direction=(-1, 0), font_size=16, parent=self.view.scene,
						 axis_color = 'black', tick_color = 'black', text_color = 'black')

		zax = scene.Axis(pos=[[0, 0], [-1, 0]], tick_direction=(0, -1), font_size=16, parent=self.view.scene,
						 axis_color='black', tick_color='black', text_color='black')
		zax.transform = scene.transforms.MatrixTransform()  # its acutally an inverted xaxis
		zax.transform.rotate(90, (0, 1, 0))  # rotate cw around yaxis
		zax.transform.rotate(-45, (0, 0, 1))  # tick direction towards (-1,-1)

		self.freeze()

	@staticmethod
	def run_app():
		vispy.app.run()
