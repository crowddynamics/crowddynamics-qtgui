import logging
import os
from collections import namedtuple

import numpy as np
import pyqtgraph as pg
from crowddynamics.core.structures.agents import is_model
from shapely.geometry import LineString
from shapely.geometry import Polygon
from crowddynamics.simulation.multiagent import MASTaskNode
from crowddynamics.io import load_config

BASE_DIR = os.path.dirname(__file__)
GRAPHICS_CFG = os.path.join(BASE_DIR, 'graphics.cfg')
GRAPHICS_CFG_SPEC = os.path.join(BASE_DIR, 'graphics_spec.cfg')


# TODO: orientation
ThreeCircle = namedtuple('ThreeCircle', 'center left right')


def frames_per_second():
    """Timer for computing frames per second"""
    from timeit import default_timer as timer

    last_time = timer()
    fps_prev = 0.0
    while True:
        now = timer()
        dt = now - last_time
        last_time = now
        fps = 1.0 / dt
        s = np.clip(3 * dt, 0, 1)
        fps_prev = fps_prev * (1 - s) + fps * s
        yield fps_prev


class GuiCommunication(MASTaskNode):
    """Communication between the GUI and simulation."""
    # TODO:


def circular(radius):
    """Defaults settings for circular plot items

    Args:
        radius (numpy.ndarray): 

    Returns:
        pyqtgraph.PlotDataItem: 
    """
    settings = {
        'pxMode': False,
        'pen': None,
        'symbol': 'o',
        'symbolSize': 2 * radius,
        # 'symbolPen': np.zeros(radius, dtype=object),
        # 'symbolBrush': np.zeros(radius, dtype=object)
    }
    return pg.PlotDataItem(**settings)


def three_circle(r_center, r_left, r_right):
    """Three circles"""
    return ThreeCircle(center=circular(r_center),
                       left=circular(r_left),
                       right=circular(r_right))


def linestring(geom, **kargs):
    """Make plotitem from LineString

    Args:
        geom (LineString|LinearRing): 

    Returns:
        PlotDataItem: 
    """
    # TODO: MultiLineString
    return pg.PlotDataItem(*geom.xy, **kargs)


def polygon(geom, **kargs):
    """Make plotitem from Polygon

    Args:
        geom (Polygon):  
    
    Returns:
        PlotDataItem:
    """
    return pg.PlotDataItem(*geom.exterior.xy, **kargs)


class MultiAgentPlot(pg.PlotItem):
    r"""MultiAgentPlot 

    GraphicsItem for displaying individual graphics of individual simulation.
    """
    logger = logging.getLogger(__name__)

    def __init__(self, parent=None):
        super(MultiAgentPlot, self).__init__(parent)
        self.setAspectLocked(lock=True, ratio=1)
        self.showGrid(x=True, y=True, alpha=0.25)
        self.disableAutoRange()

        # Utils
        # TODO: use configurations when setting plotitems
        self.configs = load_config(GRAPHICS_CFG, GRAPHICS_CFG_SPEC)
        self.fsp = frames_per_second()

        # Geometry
        self.__domain = None
        self.__obstacles = None
        self.__targets = None
        self.__agents = None

    @property
    def domain(self):
        return self.__domain

    @domain.setter
    def domain(self, geom):
        x, y = geom.exterior.xy
        x, y = np.asarray(x), np.asarray(y)
        self.setRange(xRange=(x.min(), x.max()), yRange=(y.min(), y.max()))
        item = polygon(geom)
        self.addItem(item)
        self.__domain = item

    @property
    def obstacles(self):
        return self.__obstacles

    @obstacles.setter
    def obstacles(self, geom):
        item = polygon(geom)
        self.addItem(item)
        self.__obstacles = item

    @property
    def targets(self):
        return self.__targets

    @targets.setter
    def targets(self, geom):
        item = polygon(geom)
        self.addItem(item)
        self.__targets = item

    @staticmethod
    def set_agents_data(agents, item):
        # TODO: set symbolBrushes and symbolPens through PlotDataItem.opts
        if is_model(agents, 'circular'):
            item.setData(agents['position'])
        elif is_model(agents, 'three_circle'):
            for i, a in zip(item, ('position', 'position_ls', 'position_rs')):
                i.setData(agents[a])
        else:
            raise NotImplementedError

    @property
    def agents(self):
        return self.__agents

    @agents.setter
    def agents(self, agents):
        if is_model(agents, 'circular'):
            item = circular(agents['radius'])
            self.addItem(item)
        elif is_model(agents, 'three_circle'):
            item = three_circle(agents['r_t'], agents['r_s'], agents['r_s'])
            for i in item:
                self.addItem(i)
        else:
            raise NotImplementedError

        self.set_agents_data(agents, item)
        self.__agents = item

    def configure(self, simulation):
        r"""Configure plot items

        Args:
            simulation (MultiAgentSimulation): 
        """
        # Clear previous plots and items
        self.clearPlots()
        self.clear()

        if simulation.domain:
            self.domain = simulation.domain
        if simulation.obstacles:
            self.obstacles = simulation.obstacles
        if simulation.targets:
            self.targets = simulation.targets
        self.agents = simulation.agents_array

    def update_data(self, agents):
        """Update plot data

        Args:
            agents (numpy.ndarray): 
        """
        self.set_agents_data(agents, self.agents)
        self.setTitle('%0.2f fps' % next(self.fps))
