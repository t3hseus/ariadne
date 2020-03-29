from PyQt5.QtWidgets import *
import vispy.app
import sys
import os
import numpy as np
from vispy import app, geometry
import vispy.scene
from vispy.color import Color

from vispy.scene.visuals import Polygon, Ellipse, Rectangle, RegularPolygon
from vispy import app, scene
from vispy.app import use_app
from vispy.visuals.shaders import Function
from vispy.visuals.collections import PointCollection

use_app('PyQt5')



class MyCanvas(vispy.scene.SceneCanvas):

	def __init__(self, size: (800, 500), watch_dir: str = "."):
		vispy.scene.SceneCanvas.__init__(self, keys='interactive', size=size, bgcolor=Color('white'))
		self.unfreeze()
		self.view = self.central_widget.add_view()
		self.freeze()

	def drawSome(self):
		self.unfreeze()
		# generate data
		# pos = np.random.normal(size=(100000, 3), scale=0.2)
		# # one could stop here for the data generation, the rest is just to make the
		# # data look more interesting. Copied over from magnify.py
		# centers = np.random.normal(size=(50, 3))
		# indexes = np.random.normal(size=100000, loc=centers.shape[0] / 2.,
		# 						   scale=centers.shape[0] / 3.)
		# indexes = np.clip(indexes, 0, centers.shape[0] - 1).astype(int)
		# scales = 10 ** (np.linspace(-2, 0.5, centers.shape[0]))[indexes][:, np.newaxis]
		# pos *= scales
		# pos += centers[indexes]
		#
		# # create scatter object and fill in the data
		# scatter = scene.Markers()
		# scatter.set_data(pos, edge_color=None, face_color=(1, 1, 1, .5), size=5)
		#
		# self.view.add(scatter)

		self.view.camera = 'turntable'  # or try 'arcball'

		# add a colored 3D axis for orientation
		ax = scene.Axis(pos=[[0, 0], [1, 0]], tick_direction=(0, -1), axis_color='r', tick_color='r', text_color='r',
						font_size=16, parent=self.view.scene)
		yax = scene.Axis(pos=[[0, 0], [0, 1]], tick_direction=(-1, 0), axis_color='g', tick_color='g', text_color='g',
						 font_size=16, parent=self.view.scene)

		zax = scene.Axis(pos=[[0, 0], [-1, 0]], tick_direction=(0, -1), axis_color='b', tick_color='b', text_color='b',
						 font_size=16, parent=self.view.scene)
		zax.transform = scene.transforms.MatrixTransform()  # its acutally an inverted xaxis
		zax.transform.rotate(90, (0, 1, 0))  # rotate cw around yaxis
		zax.transform.rotate(-45, (0, 0, 1))  # tick direction towards (-1,-1)
		# self.gridlines = visuals.GridLines(color=Color('gray'))
		# self.gridlines.set_gl_state('translucent', cull_face=False)
		# self.view.add(self.gridlines)
		self.freeze()


if __name__ == "__main__":
	app = QApplication(sys.argv)
	canvas = MyCanvas(size=(800, 500))
	canvas.drawSome()
	w = QMainWindow()
	widget = QWidget()
	w.setCentralWidget(widget)
	widget.setLayout(QVBoxLayout())
	widget.layout().addWidget(canvas.native)
	widget.layout().addWidget(QPushButton())
	w.show()
	vispy.app.run()
