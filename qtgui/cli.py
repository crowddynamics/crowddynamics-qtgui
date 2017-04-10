"""Extends crowddynamics commandline client with gui related commands"""
import logging
import sys
import platform
import os

import click
from PyQt4 import QtGui, QtCore
from loggingtools import setup_logging

from qtgui.main import MainWindow

BASE_DIR = os.path.dirname(__file__)
LOG_CFG = os.path.join(BASE_DIR, 'logging.yaml')


def user_info():
    logger = logging.getLogger(__name__)
    logger.info("Platform: %s", platform.platform())
    logger.info("Path: %s", sys.path[0])
    logger.info("Python: %s", sys.version[0:5])


def run_gui():
    r"""Launches the graphical user interface for visualizing simulation."""
    setup_logging(LOG_CFG)
    user_info()

    logger = logging.getLogger(__name__)
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


@click.group()
def main():
    pass


@main.command()
def run():
    """Launch gui for crowddynamics"""
    run_gui()


if __name__ == "__main__":
    main()
