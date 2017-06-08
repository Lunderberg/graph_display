import os
import warnings

import numpy as np
import scipy.interpolate

import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import matplotlib.patheffects as patheffects

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
        self.nodes = []
        self.connections = []
        self._node_lookup = {}

        self.layout = Layout()
        self.node_size = 0.05
        self.spline_points = 100
        self.box_size = 0.025
        self.convergence_threshold = 1e-5

        self._node_scatter = None
        self._reset_convergence()

    def add_node(self, node_name, color='blue',
                 text='', fontsize=16):
        if node_name in self._node_lookup:
            raise ValueError('Node "{}" already exists'.format(node_name))

        new_node = LogicalNode(self, node_name, len(self.nodes),
                               color, text, fontsize)
        self.nodes.append(new_node)
        self._node_lookup[node_name] = new_node
            
        self.layout.add_node()
        self._reset_convergence()

    def fix_x(self, node_name, x_pos):
        node = self._node_lookup[node_name]
        self.layout.fix_x(node.index, x_pos)
        self._reset_convergence()

    def fix_y(self, node_name, y_pos):
        node = self._node_lookup[node_name]
        self.layout.fix_y(node.index, y_pos)
        self._reset_convergence()

    def same_x(self, node_name_a, node_name_b):
        node_a = self._node_lookup[node_name_a]
        node_b = self._node_lookup[node_name_b]
        self.layout.same_x(node_a.index, node_b.index)
        self._reset_convergence()

    def same_y(self, node_name_a, node_name_b):
        node_a = self._node_lookup[node_name_a]
        node_b = self._node_lookup[node_name_b]
        self.layout.same_y(node_a.index, node_b.index)
        self._reset_convergence()

    def add_connection(self, origin_name, dest_name, enabled=True, weight=1.0,
                       boxed=False, **draw_props):
        origin = self._node_lookup[origin_name]
        dest = self._node_lookup[dest_name]
        conn = LogicalConnection(self, origin, dest)
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
        self.axes = axes
        self._draw_first(axes)
        self.ani = FixedFuncAnimation(axes.figure, self._update,
                                      init_func = lambda :self._draw_first(axes),
                                      interval=interval, blit=True, repeat=False)

    def stop(self):
        self.ani._stop()
        del self.ani

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

        return node_pos, connections

    def _draw_first(self, axes):
        axes.clear()

        self.axes.set_xlim(-0.1, 1.1)
        self.axes.set_ylim(-0.1, 1.1)
        self.axes.axis('off')

        return []

    def _update(self, frame_num):
        for i in range(5):
            self.layout.relax()

        node_pos, connections = self.normed_positions()

        self._check_for_convergence(node_pos, connections)

        updated = []

        for pos, log_node in zip(node_pos, self.nodes):
            updated.extend(log_node.update(self.axes, pos))

        for control_points, log_conn in zip(connections, self.connections):
            updated.extend(log_conn.update(self.axes, control_points))

        return updated

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
    def __init__(self, graph, name, index, color,
                 text, fontsize):
        self.graph = graph
        self.name = name
        self.index = index
        self.color = color
        self.text = text
        self.fontsize = fontsize

        self.ellipse = None
        self.text_box = None

    def update(self, axes, pos):
        if self.text_box is None and self.text:
            self.text_box = axes.text(
                0, 0, self.text,
                verticalalignment='center',
                horizontalalignment='center',
                color='white'
            )
            self.text_box.set_path_effects([
                patheffects.withStroke(linewidth=5, foreground='black')
            ])
            
        if self.ellipse is None:
            if self.text_box:
                text_width, text_height = self.text_size(axes)
                self.ellipse_width = np.sqrt(2)*text_width
                self.ellipse_height = np.sqrt(2)*text_height
            else:
                self.ellipse_width = self.graph.node_size
                self.ellipse_height = self.graph.node_size
            
            self.ellipse = patches.Ellipse((0,0),
                                           self.ellipse_width,
                                           self.ellipse_height,
                                           edgecolor='black',
                                           facecolor=self.color,
                                           animated=True,
                                           )
            axes.add_patch(self.ellipse)

        updated = []
            
        self.ellipse.center = pos
        updated.append(self.ellipse)
        
        if self.text_box:
            self.text_box.set_position(pos)
            updated.append(self.text_box)

        return updated

    def text_size(self, axes):
        if self.text_box is None:
            return None

        # Getting renderer from http://stackoverflow.com/a/22689498/2689797
        figure = self.text_box.figure
        if hasattr(figure.canvas, 'get_renderer'):
            renderer = figure.canvas.get_renderer()
        else:
            import io
            figure.canvas.print_pdf(io.BytesIO())
            renderer = figure._cachedRenderer

        canvas_coords = self.text_box.get_window_extent(renderer)
        data_coords = axes.transData.inverted().transform_bbox(canvas_coords)

        return (data_coords.width, data_coords.height)
        

class LogicalConnection:
    def __init__(self, graph, origin, dest):
        self.graph = graph
        self.origin = origin
        self.dest = dest
        self.box_size = 0.025

        self.enabled = True
        self.weight = 1.0
        self.draw_props = {}
        self.boxed = False

        self.rect = None
        self.spline = None
        self.arrow_head = None

    def update(self, axes, control_points):
        if self.spline is None:
            self.spline = axes.plot([0,0], [0,0], zorder=1, animated=True,
                                    **self.draw_props)[0]

        if self.boxed and self.rect is None:
            self.rect = patches.Rectangle(
                (0,0),
                self.graph.box_size, self.graph.box_size,
                animated=True, facecolor=self.draw_props['color'])
            axes.add_patch(self.rect)

        if self.arrow_head is None:
            arrowprops = dict(color = self.draw_props['color'],
                              arrowstyle = 'simple, head_width=.75, head_length=.75',
                              connectionstyle = 'arc3, rad=0',
                              shrinkA = 0,
                              shrinkB = 0,
                              animated = True)
            self.arrow_head = axes.annotate('', xy=(0,0), xycoords='data',
                                            xytext=(0,0), textcoords='data',
                                            animated=True,
                                            arrowprops=arrowprops,
                                            )

        self._adjust_to_ellipse_edge(control_points)
        spline_points = self._gen_spline(control_points)
            
        xvals = spline_points[:,0]
        yvals = spline_points[:,1]
        self.spline.set_xdata(xvals)
        self.spline.set_ydata(yvals)

        self.arrow_head.xy = (xvals[-1], yvals[-1])
        self.arrow_head.xyann = ((xvals[-2] + xvals[-1]*99)/100,
                                 (yvals[-2] + yvals[-1]*99)/100)

        updated = [self.spline, self.arrow_head]

        if self.rect:
            box_loc, box_angle = self._get_box_prop(spline_points)
            self.rect.set_transform(
                transforms.Affine2D()
                .translate(-self.box_size/2,-self.box_size/2)
                .rotate(box_angle)
                .translate(*box_loc)
                + axes.transData
            )
            updated.append(self.rect)

        return updated

    def _adjust_to_ellipse_edge(self, control_points):
        # Adjust the starting/ending point of each connection
        for center,outside in [(0,1), (-1,-2)]:
            w = [self.origin, self.dest][center].ellipse_width
            h = [self.origin, self.dest][center].ellipse_height
            
            if (control_points[outside,1] == control_points[center,1] or
                control_points[outside,0] == control_points[center,0]):
                xdiff = 0
                ydiff = 0

            else:
                q = ((control_points[outside,1] - control_points[center,1]) /
                     (control_points[outside,0] - control_points[center,0]))**2
                xdiff = np.sqrt(w*w*h*h/(4*h*h + 4*w*w*q))
                ydiff = np.sqrt(w*w*h*h/(4*w*w + 4*h*h/q))

                xdiff *= np.sign(control_points[outside,0] -
                                 control_points[center,0])
                ydiff *= np.sign(control_points[outside,1] -
                                 control_points[center,1])

            control_points[center,0] += xdiff
            control_points[center,1] += ydiff

    def _gen_spline(self, control_points):
        x = control_points[:,0]
        y = control_points[:,1]

        t = np.zeros(x.shape)
        t[1:] = np.sqrt((x[1:]-x[:-1])**2 + (y[1:]-y[:-1])**2)
        t = np.cumsum(t)

        num_spline_points = self.graph.spline_points

        # All points are identical, don't bother.
        if t[-1] == 0:
            x_spline = np.linspace(x[0],x[0],num_spline_points)
            y_spline = np.linspace(y[0],y[0],num_spline_points)

        else:
            t /= t[-1]
            nt = np.linspace(0, 1, num_spline_points)
            x_spline = scipy.interpolate.spline(t, x, nt)
            y_spline = scipy.interpolate.spline(t, y, nt)

        return np.array([x_spline,y_spline]).T

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
