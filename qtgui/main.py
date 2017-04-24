"""Main window for crowddynamics graphical user interface.

Graphical user interface and simulation graphics for crowddynamics implemented
using PyQt and pyqtgraph. Layout for the main window is created by using Qt
designer. [Hess2013]_, [Sepulveda2014]_

Design of the gui was inspired by the design of RtGraph [campagnola2012]_
"""
import logging
from collections import OrderedDict
from functools import partial
from multiprocessing import Queue

import numpy as np
import pyqtgraph as pg
from PyQt4 import QtGui, QtCore
from crowddynamics.config import import_simulation_callables
from crowddynamics.parse import parse_signature
from crowddynamics.simulation.multiagent import MultiAgentProcess, MASNode
from loggingtools import log_with

from qtgui.exceptions import CrowdDynamicsGUIException
from qtgui.graphics import MultiAgentPlot
from qtgui.ui.gui import Ui_MainWindow

logger = logging.getLogger(__name__)


class GuiCommunication(MASNode):
    """Communication between the GUI and simulation."""

    def __init__(self, simulation, queue):
        super(GuiCommunication, self).__init__(simulation)
        self.queue = queue

    def update(self, *args, **kwargs):
        self.queue.put(np.copy(self.simulation.agents_array))


@log_with()
def clear_queue(queue):
    """Clear all items from a queue"""
    while not queue.empty():
        queue.get()


@log_with()
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


@log_with()
def mkQComboBox(callback, default, values):
    """Create QComboBOx

    Args:
        callback: 
        default: 
        values: 

    Returns:
        QtGui.QComboBox: 

    """
    widget = QtGui.QComboBox()
    widget.addItems(values if values else [default])
    index = widget.findText(default)
    widget.setCurrentIndex(index)
    widget.currentIndexChanged[str].connect(callback)
    return widget


@log_with()
def mkQRadioButton(callback, default):
    """Create QRadioButton

    Args:
        callback: 
        default: 

    Returns:
        QtGui.QRadioButton: 

    """
    widget = QtGui.QRadioButton()
    widget.setChecked(default)
    widget.toggled.connect(callback)
    return widget


@log_with()
def mkQDoubleSpinBox(callback, default, values):
    """Create QDoubleSpinBox

    Args:
        callback: 
        default: 
        values: 

    Returns:
        QtGui.QDoubleSpinBox: 

    """
    widget = QtGui.QDoubleSpinBox()
    inf = float("inf")
    minimum, maximum = values if values else (None, None)
    widget.setMinimum(minimum if minimum else -inf)
    widget.setMaximum(maximum if maximum else inf)
    widget.setValue(default)
    widget.valueChanged.connect(callback)
    return widget


@log_with()
def mkQSpinBox(callback, default, values):
    """Create QSpinBox

    Args:
        callback: 
        default: 
        values: 

    Returns:
        QtGui.QSpinBox: 

    """
    widget = QtGui.QSpinBox()
    minimum, maximum = values if values else (None, None)
    widget.setMinimum(minimum if minimum else -int(10e7))
    widget.setMaximum(maximum if maximum else int(10e7))
    widget.setValue(default)
    widget.valueChanged.connect(callback)
    return widget


@log_with()
def create_data_widget(name, default, values, callback):
    """Create QWidget for setting data

    .. list-table::
       :header-rows: 1

       * - Type
         - Validation
         - Qt widget
       * - int
         - Tuple[int, int]
         - QSpinBox
       * - float
         - Tuple[float, float]
         - QDoubleSpinBox
       * - bool
         - bool
         - QRadioButton
       * - str
         - Sequence[str]
         - QComboBox

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
        return label, mkQSpinBox(callback, default, values)
    elif isinstance(default, float):
        return label, mkQDoubleSpinBox(callback, default, values)
    elif isinstance(default, bool):
        return label, mkQRadioButton(callback, default)
    elif isinstance(default, str):
        return label, mkQComboBox(callback, default, values)
    else:
        logger = logging.getLogger(__name__)
        error = CrowdDynamicsGUIException(
            'Invalid default type: {type}'.format(type=type(default)))
        logger.error(error)
        raise error


@log_with()
def configs_dict(confpath):
    """Configs dictionary
    
    ::
        
        name:
            func: Callable[MultiAgentSimulation]
            specs: Sequence[ArgSpec]
    
    Args:
        confpath: 

    Returns:

    """
    def iterable():
        for name, func in import_simulation_callables(confpath):
            specs = list(parse_signature(func))
            yield name, {'func': func,
                         'specs': specs,
                         'kwargs': OrderedDict(
                                (spec.name, spec.default) for spec in specs)}
    return OrderedDict(iterable())


class MainWindow(QtGui.QMainWindow, Ui_MainWindow):
    r"""MainWindow

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
        super(MainWindow, self).__init__()
        self.setupUi(self)

        # Simulation with multiprocessing
        self.simulation = None
        self.configs = dict()  # TODO: better configuration handling
        self.process = None
        self.queue = Queue(maxsize=4)

        # Graphics widget for plotting simulation data.
        pg.setConfigOptions(antialias=True)
        self.timer = QtCore.QTimer(self)
        self.plot = MultiAgentPlot()
        self.graphicsLayout.setBackground(None)
        self.graphicsLayout.addItem(self.plot, 0, 0)

        # Buttons
        self.initButton = QtGui.QPushButton("Initialize Simulation")

        # Sets the functionality and values for the widgets.
        self.enable_controls(False)  # Disable until simulation is set
        self.timer.timeout.connect(self.update_plots)
        self.startButton.clicked.connect(self.start)
        self.stopButton.clicked.connect(self.stop)
        self.initButton.clicked.connect(self.set_simulation)
        self.simulationsBox.currentIndexChanged[str].connect(self.set_sidebar)
        self.actionOpen.triggered.connect(self.load_simulation_cfg)

    @log_with(qualname=True, ignore=('self',))
    def enable_controls(self, boolean):
        """Enable controls

        Args:
            boolean (bool):
        """
        self.startButton.setEnabled(boolean)
        self.stopButton.setEnabled(boolean)
        self.saveButton.setEnabled(boolean)

    @log_with(qualname=True, ignore=('self',))
    def reset_buffers(self):
        r"""Reset buffers"""
        clear_queue(self.queue)

    @log_with(qualname=True, ignore=('self',))
    def set_simulation_cfg(self, confpath):
        self.configs = configs_dict(confpath)
        self.simulationsBox.addItems(list(self.configs.keys()))

    @log_with(qualname=True, ignore=('self',))
    def load_simulation_cfg(self):
        """Load simulation configurations"""
        self.simulationsBox.clear()
        confpath = QtGui.QFileDialog().getOpenFileName(
            self, 'Open file', '', 'Conf files (*.cfg)')
        self.set_simulation_cfg(confpath)

    @log_with(qualname=True, ignore=('self',))
    def _callback(self, simuname, key, value):
        self.configs[simuname]['kwargs'][key] = value

    @log_with(qualname=True, ignore=('self',))
    def set_sidebar(self, simuname):
        """Set sidebar

        Args:
            simuname (str):
        """
        self.clear_sidebar()

        for spec in self.configs[simuname]['specs']:
            label, widget = create_data_widget(
                spec.name, spec.default, spec.annotation,
                partial(self._callback, simuname, spec.name))
            self.sidebarLeft.addWidget(label)
            self.sidebarLeft.addWidget(widget)

        self.sidebarLeft.addWidget(self.initButton)

    @log_with(qualname=True, ignore=('self',))
    def clear_sidebar(self):
        r"""Clear sidebar"""
        clear_widgets(self.sidebarLeft)

    def set_simulation(self):
        r"""Set simulation"""
        # Clear data from the old simulation
        self.reset_buffers()

        # Create new simulation
        simuname = self.simulationsBox.currentText()
        cfg = self.configs[simuname]
        simulation = cfg['func'](**cfg['kwargs'])
        communication = GuiCommunication(simulation, self.queue)

        node = simulation.tasks['Reset']
        node.inject_after(communication)

        self.plot.configure(simulation)
        self.simulation = simulation

        # Last enable controls
        self.enable_controls(True)

    def stop_plotting(self):
        self.timer.stop()
        self.enable_controls(True)
        self.process = None

    def update_plots(self):
        r"""Update plots. Consumes data from the queue."""
        agents = self.queue.get()
        if agents is not MultiAgentProcess.EndProcess:
            try:
                self.plot.update_data(agents)
            except CrowdDynamicsGUIException as error:
                self.logger.error('Plotting stopped to error: {}'.format(
                    error
                ))
                self.stop_plotting()
        else:
            self.stop_plotting()

    def start(self):
        """Start simulation process and updating plot."""
        if self.simulation:
            # Wrap the simulation into a process class here because we can
            # only use processes once.
            self.startButton.setEnabled(False)
            self.process = MultiAgentProcess(self.simulation, self.queue)
            self.process.start()
            self.timer.start(1)
        else:
            self.logger.info("Simulation is not set.")

    def stop(self):
        """Stops simulation process and updating the plot"""
        if self.process:
            self.process.stop()
        else:
            self.logger.info("There are no processes running.")
