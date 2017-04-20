"""Extends crowddynamics commandline client with gui related commands"""
import logging
import os
import sys

import click
from PyQt4 import QtGui, QtCore
from crowddynamics.logging import setup_logging

from qtgui.main import MainWindow

BASE_DIR = os.path.dirname(__file__)
LOG_CFG = os.path.join(BASE_DIR, 'logging.yaml')


def run_gui(simulation_cfg=None):
    r"""Launches the graphical user interface for visualizing simulation."""
    setup_logging(log_cfg=LOG_CFG)

    logger = logging.getLogger(__name__)
    logger.info('Starting GUI')
    app = QtGui.QApplication(sys.argv)
    win = MainWindow()
    if simulation_cfg:
        win.set_simulation_cfg(simulation_cfg)
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
@click.option('--simulation_cfg', type=str, default=None)
def run(simulation_cfg):
    """Launch gui for crowddynamics"""
    run_gui(simulation_cfg)


if __name__ == "__main__":
    main()
