import os
import sys

import PyQt5
from PyQt5.QtCore import QObject, Qt, QUrl, QSize, qInstallMessageHandler
from PyQt5.QtQuick import QQuickView
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QApplication, QSizePolicy

from canvas import MyCanvas

import pathlib

from visualizer import Visualizer


def qt_message_handler(mode, context, message):
	if mode == PyQt5.QtCore.QtInfoMsg:
		mode = 'INFO'
	elif mode == PyQt5.QtCore.QtWarningMsg:
		mode = 'WARNING'
	elif mode == PyQt5.QtCore.QtCriticalMsg:
		mode = 'CRITICAL'
	elif mode == PyQt5.QtCore.QtFatalMsg:
		mode = 'FATAL'
	else:
		mode = 'DEBUG'
	fileName = context.file
	if fileName is None:
		fileName = ""
	print('[UI_DEBUG][%s]: %s: %d:: %s' % (mode, fileName.rsplit('/')[-1], context.line, message))

if __name__ == "__main__":
	sys_argv = sys.argv
	sys_argv += ['--style', 'imagine']
	app = QApplication(sys.argv)
	qInstallMessageHandler(qt_message_handler)
	win = QMainWindow()
	win.setStyleSheet("background-color:white")
	widget = QWidget(win)

	win.setCentralWidget(widget)
	layout = QVBoxLayout()
	layout.setSpacing(0)
	layout.setContentsMargins(0,0,0,0)
	widget.setLayout(layout)

	#qml view
	sidePanelView = QQuickView()
	sidePanelView.setResizeMode(QQuickView.SizeRootObjectToView)
	qmlwidget = QWidget.createWindowContainer(sidePanelView, win)
	qmlwidget.setFocusPolicy(Qt.WheelFocus)
	#qmlwidget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

	def on_qml_loaded(a):
		if a != QQuickView.Ready:
			return

		qmlwidget.setFixedHeight(80)

		widget.layout().addWidget(qmlwidget)

		canvas = MyCanvas(size=(400, 500))
		canvas.build_axes()

		basic_config = {
			"cache_path": "",
			"mode": "3d"
		}
		vis = Visualizer(basic_config, win)
		vis.setup_from_vispy_pyqt5(win, canvas, sidePanelView)

		widget.layout().addWidget(canvas.native)
		win.show()
		canvas.run_app()

	sidePanelView.statusChanged.connect(on_qml_loaded)
	sidePanelView.setSource(QUrl.fromLocalFile(os.path.dirname(os.path.abspath(__file__)) + "/../UI/TopPanel.qml"))
	app.exec_()
