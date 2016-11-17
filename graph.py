import numpy as np
import matplotlib.animation
from matplotlib.collections import EllipseCollection
import matplotlib.pyplot as plt

from layout import Layout

class Graph:
    def __init__(self):
        self._nodes = {}
        self.connections = []

        self.layout = Layout()
        self._connection_lines = []
        self._node_scatter = None

    def add_node(self, node_name):
        if node_name in self._nodes:
            raise ValueError('Node "{}" already exists'.format(node_name))

        self._nodes[node_name] = LogicalNode(node_name, len(self._nodes))
        self.layout.add_node()

    def add_connection(self, origin_name, dest_name, enabled=True, weight=1.0):
        origin = self._nodes[origin_name]
        dest = self._nodes[dest_name]
        conn = LogicalConnection(origin, dest)
        conn.enabled = enabled
        conn.weight = weight
        self.connections.append(conn)

        self.layout.add_connection(origin.index, dest.index)

    @property
    def nodes(self):
        return self._nodes.values()

    def draw(self, axes, interval=25):
        self._draw_first(axes)
        self.ani = matplotlib.animation.FuncAnimation(axes.figure, self._update,
                                                      init_func = lambda :self._draw_first(axes),
                                                      interval=interval, blit=False)

    def _draw_first(self, axes):
        axes.clear()
        self._connection_lines.clear()
        self._node_scatter = None

        node_pos, conn_origin, conn_dest = self.layout.positions()

        for origin,dest in zip(conn_origin, conn_dest):
            uniform = np.arange(0, 1, 0.01)

            # Shape to match.
            origin = origin.reshape((2,1))
            dest = dest.reshape((2,1))
            uniform = uniform.reshape((1,len(uniform)))

            pos = origin + (dest-origin)*uniform

            line = axes.plot(pos[0], pos[1],
                             zorder=1)
            self._connection_lines.append(line)


        self._node_scatter = EllipseCollection(
            offsets=node_pos,
            widths=0.05, heights=0.05, angles=0, units='xy',
            facecolors='blue', edgecolor='black',
            zorder=2,
            transOffset=axes.transData)
        axes.add_collection(self._node_scatter)

        axes.set_xlim(-0.1, 1.1)
        axes.set_ylim(-0.1, 1.1)
        axes.axis('off')

    def _update(self, *args):
        self.layout.relax()

        node_pos, conn_origin, conn_dest = self.layout.positions()

        for line,origin,dest in zip(self._connection_lines,conn_origin,conn_dest):
            uniform = np.arange(0, 1, 0.01)

            origin = origin.reshape((2,1))
            dest = dest.reshape((2,1))
            uniform = uniform.reshape((1,len(uniform)))

            pos = origin + (dest-origin)*uniform

            line[0].set_xdata(pos[0])
            line[0].set_ydata(pos[1])

        self._node_scatter.set_offsets(node_pos)


class LogicalNode:
    def __init__(self, name, index):
        self.name = name
        self.index = index

class LogicalConnection:
    def __init__(self, origin, dest):
        self.origin = origin
        self.dest = dest
        self.enabled = True
        self.weight = 1.0
