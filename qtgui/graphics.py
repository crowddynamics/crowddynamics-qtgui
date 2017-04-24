import logging
import os
from collections import Iterable

import numba
import numpy as np
import pyqtgraph as pg
from crowddynamics.config import load_config
from crowddynamics.core.structures.agents import is_model
from crowddynamics.core.vector.vector2D import normalize, unit_vector

from loggingtools import log_with
from numba import f8
from shapely.geometry import Point, LineString, Polygon

from qtgui.exceptions import CrowdDynamicsGUIException, FeatureNotImplemented

CONFIG_DIR = os.path.join(os.path.dirname(__file__), 'conf')
GRAPHICS_CFG = os.path.join(CONFIG_DIR, 'graphics.cfg')
GRAPHICS_CFG_SPEC = os.path.join(CONFIG_DIR, 'graphics_spec.cfg')


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


def circles(radius, **kargs):
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
    return pg.PlotDataItem(**settings, **kargs)


@numba.jit([(f8[:, :], f8[:, :], f8[:])],
           nopython=True, nogil=True, cache=True)
def lines(origin, direction, length):
    """Lines

    Args:
        origin (numpy.ndarray): 
        direction (numpy.ndarray): 
        length (numpy.ndarray): 
    """
    n, m = origin.shape
    values = np.empty(shape=(2 * n, m))
    for i in range(n):
        values[2 * i, :] = origin[i, :]
        values[2 * i + 1, :] = origin[i, :] + normalize(direction[i]) * length[i]
    return values


def lines_connect(n):
    connect = np.ones(shape=2 * n, dtype=np.uint8)
    connect[1::2] = 0
    return connect


class AgentsBase(object):
    __slots__ = ('center', 'left', 'right', 'orientation', 'direction',
                 'target_direction')

    def __init__(self):
        self.center = None
        self.left = None
        self.right = None
        self.orientation = None
        self.direction = None
        self.target_direction = None

    def addItem(self, widget: pg.PlotItem):
        raise NotImplementedError

    def setData(self, agents: np.ndarray):
        raise NotImplementedError


class CircularAgents(AgentsBase):
    def __init__(self, agents):
        super().__init__()
        assert is_model(agents, 'circular'), \
            'Agent should be circular model'
        connect = lines_connect(agents.size)
        self.center = circles(agents['radius'])
        self.direction = pg.PlotDataItem(connect=connect)
        self.target_direction = pg.PlotDataItem(connect=connect)

    def addItem(self, widget: pg.PlotItem):
        widget.addItem(self.center)
        widget.addItem(self.direction)
        widget.addItem(self.target_direction)

    def setData(self, agents):
        self.center.setData(agents['position'])
        self.direction.setData(
            lines(agents['position'], agents['velocity'], 2 * agents['radius']))
        self.target_direction.setData(
            lines(agents['position'], agents['target_direction'],
                  2 * agents['radius']))


class ThreeCircleAgents(AgentsBase):
    def __init__(self, agents):
        super().__init__()
        assert is_model(agents, 'three_circle'), \
            'Agent should the three_circle model'
        self.center = circles(agents['r_t'])
        self.left = circles(agents['r_s'])
        self.right = circles(agents['r_s'])
        connect = lines_connect(agents.size)
        self.orientation = pg.PlotDataItem(connect=connect)
        self.direction = pg.PlotDataItem(connect=connect)
        self.target_direction = pg.PlotDataItem(connect=connect)

    def addItem(self, widget: pg.PlotItem):
        widget.addItem(self.center)
        widget.addItem(self.left)
        widget.addItem(self.right)
        widget.addItem(self.orientation)
        widget.addItem(self.direction)
        widget.addItem(self.target_direction)

    def setData(self, agents):
        self.center.setData(agents['position'])
        self.left.setData(agents['position_ls'])
        self.right.setData(agents['position_rs'])
        self.orientation.setData(
            lines(agents['position'], unit_vector(agents['orientation']),
                  1.1 * agents['radius']))
        self.direction.setData(
            lines(agents['position'], agents['velocity'], 2 * agents['radius']))
        self.target_direction.setData(
            lines(agents['position'], agents['target_direction'],
                  2 * agents['radius']))


@log_with()
def linestring(geom, **kargs):
    """Make plotitem from LineString

    Args:
        geom (LineString|LinearRing): 

    Returns:
        PlotDataItem: 
    """
    # TODO: MultiLineString
    return pg.PlotDataItem(*geom.xy, **kargs)


@log_with()
def polygon(geom, **kargs):
    """Make plotitem from Polygon

    Args:
        geom (Polygon):  
    
    Returns:
        PlotDataItem:
    """
    return pg.PlotDataItem(*geom.exterior.xy, **kargs)


@log_with()
def shapes(geom, **kargs):
    """Shape

    Args:
        geom: 
        **kargs: 

    Returns:
        list:
    """
    if isinstance(geom, Point):
        return []  # NotImplemented
    if isinstance(geom, LineString):
        return [linestring(geom, **kargs)]
    elif isinstance(geom, Polygon):
        return [polygon(geom, **kargs)]
    elif isinstance(geom, Iterable):
        return sum((shapes(geo) for geo in geom), [])
    else:
        raise CrowdDynamicsGUIException


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
        self.configs = load_config(GRAPHICS_CFG, GRAPHICS_CFG_SPEC)
        self.fps = frames_per_second()

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
        """Set domain

        Args:
            geom (Polygon): 
        """
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
        """Set obstacles

        Args:
            geom (LineString|MultiLineString): 
        """
        items = shapes(geom)
        for item in items:
            self.addItem(item)
        self.__obstacles = items

    @property
    def targets(self):
        return self.__targets

    @targets.setter
    def targets(self, geom):
        """Targets

        Args:
            geom (Polygon|MultiPolygon): 
        """
        items = shapes(geom)
        for item in items:
            self.addItem(item)
        self.__targets = items

    @property
    def agents(self):
        return self.__agents

    @agents.setter
    def agents(self, agents):
        if is_model(agents, 'circular'):
            self.__agents = CircularAgents(agents)
            self.agents.addItem(self)
            self.agents.setData(agents)
        elif is_model(agents, 'three_circle'):
            self.__agents = ThreeCircleAgents(agents)
            self.agents.addItem(self)
            self.agents.setData(agents)
        else:
            raise FeatureNotImplemented('Wrong agents type: "{}"'.format(
                agents))

    @log_with(qualname=True, timed=True, ignore=('self',))
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
        self.agents.setData(agents)
        self.setTitle('%0.2f fps' % next(self.fps))
