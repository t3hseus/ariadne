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
	app.setOrganizationName("theseus")
	app.setOrganizationDomain("ariadne")
	app.setApplicationName("visualizer")
	win = QMainWindow()
	win.setStyleSheet("background-color:white")
	widget = QWidget(win)

	win.setCentralWidget(widget)
	layout = QVBoxLayout()
	layout.setSpacing(0)
	layout.setContentsMargins(0,0,0,0)
	widget.setLayout(layout)

	#qml top panel view
	top_panel_view = QQuickView()
	top_panel_view.setResizeMode(QQuickView.SizeRootObjectToView)
	qml_top_panel_widget = QWidget.createWindowContainer(top_panel_view, win)
	qml_top_panel_widget.setFocusPolicy(Qt.WheelFocus)

	#qml bot panel view
	bot_panel_view = QQuickView()
	bot_panel_view.setResizeMode(QQuickView.SizeRootObjectToView)
	qml_bot_panel_widget = QWidget.createWindowContainer(bot_panel_view, win)
	qml_bot_panel_widget.setFocusPolicy(Qt.WheelFocus)

	basic_config = {
		"cache_path": "",
		"mode": "3d"
	}
	vis = Visualizer(basic_config, win)

	canvas = MyCanvas(size=(400, 500))
	canvas.build_axes()
	vis.setup_from_vispy_pyqt5(win, canvas, top_panel_view)

	def on_qml_loaded(a):
		if a != QQuickView.Ready:
			return

		qml_top_panel_widget.setFixedHeight(140)
		qml_bot_panel_widget.setFixedHeight(200)
		widget.layout().addWidget(qml_top_panel_widget)
		widget.layout().addWidget(canvas.native)
		widget.layout().addWidget(qml_bot_panel_widget)

		win.show()
		canvas.run_app()
		exit()


	bot_panel_view.setSource(QUrl.fromLocalFile(os.path.dirname(os.path.abspath(__file__)) + "/../UI/BotPanel.qml"))

	top_panel_view.statusChanged.connect(on_qml_loaded)
	top_panel_view.setSource(QUrl.fromLocalFile(os.path.dirname(os.path.abspath(__file__)) + "/../UI/TopPanel.qml"))



	assert False, "You should not enter this, something went wrong."
