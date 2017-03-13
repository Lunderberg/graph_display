import os
import warnings

import numpy as np
import scipy.interpolate

import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection
import matplotlib.patches as patches
import matplotlib.transforms as transforms


try:
    from clayout import Layout
except ImportError:
    warnings.warn(
        'Compiled version not found, falling back to pure python\n' +
        'Run "scons" in {} for more speed'.format(os.path.dirname(__file__))
    )
    from layout import Layout

from fixed_func_animation import FixedFuncAnimation


class Graph:
    def __init__(self):
        self._nodes = {}
        self.connections = []

        self.layout = Layout()
        self.node_size = 0.05
        self.spline_points = 100
        self.box_size = 0.025
        self.convergence_threshold = 1e-5

        self._connection_lines = []
        self._arrow_heads = []
        self._node_scatter = None
        self._reset_convergence()

    def add_node(self, node_name, color='blue'):
        if node_name in self._nodes:
            raise ValueError('Node "{}" already exists'.format(node_name))

        self._nodes[node_name] = LogicalNode(node_name, len(self._nodes), color)
        self.layout.add_node()
        self._reset_convergence()

    def fix_x(self, node_name, x_pos):
        node = self._nodes[node_name]
        self.layout.fix_x(node.index, x_pos)
        self._reset_convergence()

    def fix_y(self, node_name, y_pos):
        node = self._nodes[node_name]
        self.layout.fix_y(node.index, y_pos)
        self._reset_convergence()

    def same_x(self, node_name_a, node_name_b):
        node_a = self._nodes[node_name_a]
        node_b = self._nodes[node_name_b]
        self.layout.same_x(node_a.index, node_b.index)
        self._reset_convergence()

    def same_y(self, node_name_a, node_name_b):
        node_a = self._nodes[node_name_a]
        node_b = self._nodes[node_name_b]
        self.layout.same_y(node_a.index, node_b.index)
        self._reset_convergence()

    def add_connection(self, origin_name, dest_name, enabled=True, weight=1.0,
                       boxed=False, **draw_props):
        origin = self._nodes[origin_name]
        dest = self._nodes[dest_name]
        conn = LogicalConnection(origin, dest)
        conn.enabled = enabled
        conn.weight = weight
        conn.boxed = boxed

        if 'color' not in draw_props:
            draw_props['color'] = 'black'
        conn.draw_props = draw_props
        self.connections.append(conn)

        self.layout.add_connection(origin.index, dest.index)
        self._reset_convergence()

    def draw(self, axes, interval=25):
        #self._draw_first(axes)
        self.ani = FixedFuncAnimation(axes.figure, self._update,
                                      init_func = lambda :self._draw_first(axes),
                                      interval=interval, blit=True, repeat=False)

    def _reset_convergence(self):
        # Currently, too much work for something that isn't used.  The
        # _draw_first and _update functions would need to be merged,
        # so that new nodes can be added.  In addition, if the layout
        # has already converged, the animation needs to be restarted.
        # After calling FuncAnimation._stop(), calls to _start() do
        # not work.  This may be adjustable by overriding _stop and
        # commenting out the removal of event_source.  This does raise
        # other issues, though, as then the reference is not dropped
        # when the canvas is closed.
        if hasattr(self,'ani'):
            raise NotImplementedError('Cannot modify graph after display')

        self.prev_positions = None
        self.converged = False


    def normed_positions(self):
        node_pos, connections = self.layout.positions()

        xmin = connections[:,:,0].min()
        xmax = connections[:,:,0].max()
        ymin = connections[:,:,1].min()
        ymax = connections[:,:,1].max()

        node_pos[:,0] = (node_pos[:,0] - xmin)/(xmax-xmin)
        node_pos[:,1] = (node_pos[:,1] - ymin)/(ymax-ymin)
        connections[:,:,0] = (connections[:,:,0] - xmin)/(xmax-xmin)
        connections[:,:,1] = (connections[:,:,1] - ymin)/(ymax-ymin)

        w = self.node_size
        h = self.node_size

        # Adjust the starting/ending point of each connection
        for center,outside in [(0,1), (-1,-2)]:
            with np.errstate(invalid='ignore'):
                q = ((connections[:,outside,1] - connections[:,center,1]) /
                     (connections[:,outside,0] - connections[:,center,0]))**2
            xdiff = np.sqrt(w*w*h*h/(4*h*h + 4*w*w*q))
            ydiff = np.sqrt(w*w*h*h/(4*w*w + 4*h*h/q))

            xdiff *= np.sign(connections[:,outside,0] - connections[:,center,0])
            ydiff *= np.sign(connections[:,outside,1] - connections[:,center,1])

            xdiff[np.isnan(xdiff)] = 0
            ydiff[np.isnan(ydiff)] = 0

            connections[:,center,0] += xdiff
            connections[:,center,1] += ydiff

        return node_pos, connections


    def _gen_splines(self, connections):
        for control_points in connections:
            x = control_points[:,0]
            y = control_points[:,1]

            t = np.zeros(x.shape)
            t[1:] = np.sqrt((x[1:]-x[:-1])**2 + (y[1:]-y[:-1])**2)
            t = np.cumsum(t)

            # All points are identical, don't bother.
            if t[-1] == 0:
                x_spline = np.linspace(x[0],x[0],self.spline_points)
                y_spline = np.linspace(y[0],y[0],self.spline_points)

            else:
                t /= t[-1]
                nt = np.linspace(0, 1, self.spline_points)
                x_spline = scipy.interpolate.spline(t, x, nt)
                y_spline = scipy.interpolate.spline(t, y, nt)

            yield np.array([x_spline,y_spline]).T

    def _draw_first(self, axes):
        axes.clear()
        self._connection_lines.clear()
        self._node_scatter = None

        node_pos, connections = self.normed_positions()

        self._connection_lines = []
        self._arrow_heads = []

        opt = dict(color = 'black',
                   arrowstyle = 'simple, head_width=.75, head_length=.75',
                   connectionstyle = 'arc3, rad=0',
                   shrinkA = 0,
                   shrinkB = 0,
                   animated = True)

        for log_conn,spline in zip(self.connections,self._gen_splines(connections)):
            xvals = spline[:,0]
            yvals = spline[:,1]

            # Draw the spline itself
            self._connection_lines.append(
                axes.plot(xvals, yvals, zorder=1,  animated=True, **log_conn.draw_props)[0])
            # Arrow at the end of the spline
            self._arrow_heads.append(
                axes.annotate('', xy=(xvals[-1], yvals[-1]), xycoords='data',
                              xytext=((xvals[-2] + xvals[-1]*99)/100,
                                      (yvals[-2] + yvals[-1]*99)/100),
                              textcoords='data',
                              animated=True,
                              arrowprops=opt))

            # Box in the middle of the spline.
            if log_conn.boxed:
                box_loc, box_angle = self._get_box_prop(spline)
                rect = patches.Rectangle((0,0), self.box_size, self.box_size, animated=True,
                                         facecolor=log_conn.draw_props['color'])
                rect.set_transform(transforms.Affine2D()
                                   .translate(-self.box_size/2,-self.box_size/2)
                                   .rotate(box_angle)
                                   .translate(*box_loc) + axes.transData)
                log_conn.rect = rect
                axes.add_patch(rect)

        nodecolors = [node.color for node in sorted(self._nodes.values(),key=lambda node:node.index)]
        self._node_scatter = EllipseCollection(
            offsets=node_pos,
            widths=self.node_size, heights=self.node_size, angles=0, units='xy',
            facecolors=nodecolors, edgecolor='black',
            zorder=2,
            transOffset=axes.transData, animated=True)
        axes.add_collection(self._node_scatter)

        axes.set_xlim(-0.1, 1.1)
        axes.set_ylim(-0.1, 1.1)
        axes.axis('off')

        return self._connection_lines + self._arrow_heads + [self._node_scatter]

    def _get_box_prop(self,spline):
        npoints = spline.shape[0]
        if npoints % 2 == 0:
            a = spline[npoints//2]
            b = spline[npoints//2 + 1]
            box_loc = (a+b)/2.0
            box_angle = np.arctan2(b[1]-a[1],b[0]-a[0])
        else:
            a = spline[npoints//2 - 1]
            b = spline[npoints//2]
            c = spline[npoints//2 + 1]
            box_loc = b
            box_angle = np.arctan2(c[1]-a[1],c[0]-a[0])

        return box_loc, box_angle

    def _update(self, frame_num):
        for i in range(5):
            self.layout.relax()

        node_pos, connections = self.normed_positions()

        self._check_for_convergence(node_pos, connections)

        self._node_scatter.set_offsets(node_pos)

        boxes = []

        for line, arrow_head, spline, log_conn in zip(self._connection_lines, self._arrow_heads,
                                                      self._gen_splines(connections), self.connections):
            xvals = spline[:,0]
            yvals = spline[:,1]
            line.set_xdata(xvals)
            line.set_ydata(yvals)
            arrow_head.xy = (xvals[-1], yvals[-1])
            arrow_head.xyann = ((xvals[-2] + xvals[-1]*99)/100,
                                (yvals[-2] + yvals[-1]*99)/100)

            if log_conn.rect:
                axes = log_conn.rect.axes
                box_loc, box_angle = self._get_box_prop(spline)
                log_conn.rect.set_transform(
                    transforms.Affine2D()
                    .translate(-self.box_size/2,-self.box_size/2)
                    .rotate(box_angle)
                    .translate(*box_loc)
                    + axes.transData
                    )

                boxes.append(log_conn.rect)

        return self._connection_lines + self._arrow_heads + [self._node_scatter] + boxes

    def _check_for_convergence(self, node_pos, connections):
        if self.prev_positions is not None:
            prev_node_pos, prev_connections = self.prev_positions
            max_change = max(np.abs(node_pos - prev_node_pos).max(),
                             np.abs(connections - prev_connections).max())
            if max_change < self.convergence_threshold:
                self.converged = True
                self.ani._stop()

        self.prev_positions = node_pos, connections


class LogicalNode:
    def __init__(self, name, index, color):
        self.name = name
        self.index = index
        self.color = color

class LogicalConnection:
    def __init__(self, origin, dest):
        self.origin = origin
        self.dest = dest
        self.enabled = True
        self.weight = 1.0
        self.draw_props = {}
        self.boxed = False
        self.rect = None
