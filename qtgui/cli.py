"""Extends crowddynamics commandline client with gui related commands"""
import logging
import sys

from crowddynamics.cli import main
from PyQt4 import QtGui, QtCore

from qtgui.main import MainWindow

logger = logging.getLogger(__name__)


def run_gui():
    r"""Launches the graphical user interface for visualizing simulation."""
    logger.info('Starting GUI')
    app = QtGui.QApplication(sys.argv)
    win = MainWindow()
    win.show()

    # Start Qt event loop unless running in interactive mode or using pyside.
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        app.exec()
    else:
        logger.warning("Interactive mode and pyside are not supported.")

    logging.info('Exiting GUI')
    logging.shutdown()

    win.close()
    app.exit()
    sys.exit()


@main()
def gui():
    """Launch gui for crowddynamics"""
    run_gui()
