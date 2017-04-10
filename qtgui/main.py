"""Main window for crowddynamics graphical user interface.

Graphical user interface and simulation graphics for crowddynamics implemented
using PyQt and pyqtgraph. Layout for the main window is created by using Qt
designer. [Hess2013]_, [Sepulveda2014]_

Design of the gui was inspired by the design of RtGraph [campagnola2012]_

"""
import logging
import sys
from multiprocessing import Queue

import pyqtgraph as pg
from PyQt4 import QtGui, QtCore
from crowddynamics.parse import ArgSpec

from qtgui.exceptions import CrowdDynamicsGUIException
from qtgui.graphics import MultiAgentPlot
from qtgui.ui.gui import Ui_MainWindow


def clear_queue(queue):
    """Clear all items from a queue"""
    while not queue.empty():
        queue.get()


def clear_widgets(layout):
    """Clear widgets from a layout
    
    Args:
        layout: 

    References
        - http://stackoverflow.com/questions/4528347/clear-all-widgets-in-a-layout-in-pyqt
    """
    for i in reversed(range(layout.count())):
        if i in (0, 1):
            continue
        layout.itemAt(i).widget().setParent(None)


def create_widget(name, default, values, callback):
    """Create QWidget for setting data

    Args:
        name (str): 
            Name for the label of the widget
        default (int|float|bool|str): 
            Default value for the widget
        values (typing.Sequence): 
            Values that are valid input for the widget
        callback (typing.Callable): 
            Callback function that is called when the value of widget changes.

    Returns:
        typing.Tuple[QtGui.QLabel, QtGui.QWidget]: 
    """
    label = QtGui.QLabel(name)

    if isinstance(default, int):
        widget = QtGui.QSpinBox()

        if values[0] is not None:
            widget.setMinimum(values[0])
        else:
            widget.setMinimum(-100000)

        if values[1] is not None:
            widget.setMaximum(values[1])
        else:
            widget.setMaximum(100000)

        widget.setValue(default)
        widget.valueChanged.connect(callback)
    elif isinstance(default, float):
        widget = QtGui.QDoubleSpinBox()

        inf = float("inf")
        if values[0] is not None:
            widget.setMinimum(values[0])
        else:
            widget.setMinimum(-inf)

        if values[1] is not None:
            widget.setMaximum(values[1])
        else:
            widget.setMaximum(inf)

        widget.setValue(default)
        widget.valueChanged.connect(callback)
    elif isinstance(default, bool):
        widget = QtGui.QRadioButton()
        widget.setChecked(default)
        widget.toggled.connect(callback)
    elif isinstance(default, str):
        widget = QtGui.QComboBox()
        widget.addItems(values)
        index = widget.findText(default)
        widget.setCurrentIndex(index)
        widget.currentIndexChanged[str].connect(callback)
    else:
        raise CrowdDynamicsGUIException('Invalid type for sidebar.')

    return label, widget


class MainWindow(QtGui.QMainWindow, Ui_MainWindow):
    r"""
    Main window for the grahical user interface. Layout is created by using
    qtdesigner and the files can be found in the *designer* folder. Makefile
    to generate python code from the designer files can be used with command::

       make gui

    Main window consists of

    - Menubar (top)
    - Sidebar (left)
    - Graphics layout widget (middle)
    - Control bar (bottom)
    """
    logger = logging.getLogger(__name__)

    def __init__(self):
        r"""
        MainWindow
        """
        super(MainWindow, self).__init__()
        self.setupUi(self)

        # Simulation with multiprocessing
        self.queue = Queue(maxsize=4)
        self.process = None

        # Graphics
        self.timer = QtCore.QTimer(self)
        self.plot = None

        # Buttons
        # self.savingButton = QtGui.QRadioButton("Save")
        self.initButton = QtGui.QPushButton("Initialize Simulation")

        # Graphics widget for plotting simulation data.
        pg.setConfigOptions(antialias=True)
        self.graphicsLayout.setBackground(None)
        self.plot = MultiAgentPlot()
        self.graphicsLayout.addItem(self.plot, 0, 0)

        # Sets the functionality and values for the widgets.
        self.timer.timeout.connect(self.update_plots)
        self.startButton.clicked.connect(self.start)
        self.stopButton.clicked.connect(self.stop)
        self.initButton.clicked.connect(self.set_simulation)

        self.enable_controls(False)  # Disable until simulation is set

        # Menus
        names = ()  # NotImplemented
        self.simulationsBox.addItem("")  # No simulation. Clear sidebar.
        self.simulationsBox.addItems(names)
        self.simulationsBox.currentIndexChanged[str].connect(self.set_sidebar)

        # Do not use multiprocessing in windows because of different semantics
        # compared to linux.
        self.enable_multiprocessing = not sys.platform.startswith('Windows')

    def enable_controls(self, boolean):
        """Enable controls

        Args:
            boolean (bool):
        """
        self.startButton.setEnabled(boolean)
        self.stopButton.setEnabled(boolean)
        self.saveButton.setEnabled(boolean)

    def reset_buffers(self):
        r"""Reset buffers"""
        clear_queue(self.queue)

    def set_sidebar(self, name):
        """Set sidebar

        Args:
            name (str):

        """
        self.clear_sidebar()

        if name == "":
            return

        specs: ArgSpec = NotImplemented
        for name, default, annotation in specs:
            callback = NotImplemented
            label, widget = create_widget(name, default, annotation, callback)
            self.sidebarLeft.addWidget(label)
            self.sidebarLeft.addWidget(widget)

        # self.sidebarLeft.addWidget(self.savingButton)
        self.sidebarLeft.addWidget(self.initButton)

    def clear_sidebar(self):
        r"""Clear sidebar"""
        clear_widgets(self.sidebarLeft)

    def set_simulation(self):
        r"""Set simulation"""
        self.reset_buffers()

        name = self.simulationsBox.currentText()

        self.enable_controls(True)
        self.plot.configure(self.process)

        # TODO: simulation Communication

    def update_plots(self):
        r"""Update plots"""
        data = self.queue.get()
        if data:
            if not self.enable_multiprocessing:
                self.process.update()  # Sequential processing
            self.plot.update_data(data)
        else:
            self.timer.stop()
            self.enable_controls(False)
            self.process = None
            self.reset_buffers()

    def start(self):
        """Start simulation process and updating plot."""
        self.startButton.setEnabled(False)
        if self.process is not None:
            if self.enable_multiprocessing:
                self.process.start()
            else:
                self.process.update()

            self.timer.start(1)
        else:
            self.logger.info("Process is not set")

    def stop(self):
        """Stops simulation process and updating the plot"""
        if self.process is not None:
            if self.enable_multiprocessing:
                self.process.stop()
            else:
                self.queue.put(None)
        else:
            self.logger.info("Process is not set")
