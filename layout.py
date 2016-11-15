import itertools
import random

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.animation
from matplotlib.collections import EllipseCollection

import numpy as np

class Layout:
    def __init__(self, graph):
        self.graph = graph
        self.nodes = {}
        self.conditions = []
        self.spring_constant = 0.01
        self.repulsion_constant = 0.01
        self.pseudo_gravity = 0.05

        self._connection_lines = []
        self._node_scatter = None

        self.gen_nodes()

    def gen_nodes(self):
        for node in self.graph.nodes:
            if node.name not in self.nodes:
                layout_node = LayoutNode()
                layout_node.x = random.uniform(0,1)
                layout_node.y = random.uniform(0,1)
                self.nodes[node.name] = layout_node

        graph_nodes = set(node.name for node in self.graph.nodes)
        current_nodes = set(self.nodes)
        to_delete = graph_nodes - current_nodes
        for name in to_delete:
            del self.nodes[name]

    def randomize_layout(self):
        for node in self.nodes.values():
            node.x = random.uniform(0,1)
            node.y = random.uniform(0,1)

    def add_condition(self, condition):
        self.conditions.append(condition)

    def relax(self, n_iter=1, conditions=None,
              spring_constant=None, repulsion_constant=None, pseudo_gravity=None):

        if spring_constant is None:
            spring_constant = self.spring_constant

        if repulsion_constant is None:
            repulsion_constant = self.repulsion_constant

        if pseudo_gravity is None:
            pseudo_gravity = self.pseudo_gravity

        all_conditions = all_conditions = itertools.chain(
            self.conditions, conditions if conditions is not None else [])


        for _ in range(n_iter):
            forces = {name:np.array([0,0], dtype='float64') for name in self.nodes}

            # Electrostatic repulsion
            for (node_a, node_b) in itertools.combinations(self.nodes, 2):
                disp = self.nodes[node_b].pos - self.nodes[node_a].pos
                dist2 = np.dot(disp, disp)
                unit_vec = disp/np.sqrt(dist2)
                force = repulsion_constant * unit_vec / dist2
                forces[node_a] -= force
                forces[node_b] += force

            # Spring attraction
            for conn in self.graph.connections:
                from_name = conn.origin.name
                to_name = conn.dest.name
                disp = self.nodes[to_name].pos - self.nodes[from_name].pos
                force = -spring_constant * disp

                forces[to_name] += force
                forces[from_name] -= force

            # Pseudo-gravity, constant force toward zero
            for name in self.nodes:
                direction = self.nodes[name].pos
                disp = np.copy(self.nodes[name].pos)
                forces[name] -= pseudo_gravity/(1 + np.exp(-disp))

            # Update node positions
            for name,node in self.nodes.items():
                node.pos += forces[name]

            # Apply conditions
            for condition in all_conditions:
                self._apply_condition(condition)

    def _apply_condition(self, condition):
        if condition[0] == 'fixed_x':
            self.nodes[condition[1]].x = condition[2]

        elif condition[0] == 'fixed_y':
            self.nodes[condition[1]].y = condition[2]

        elif condition[0] == 'same_x':
            new_x = sum(self.nodes[node_name].x for node_name in condition[1])/len(condition[1])
            for node_name in condition[1]:
                self.nodes[node_name].x = new_x

        elif condition[0] == 'same_y':
            new_y = sum(self.nodes[node_name].y for node_name in condition[1])/len(condition[1])
            for node_name in condition[1]:
                self.nodes[node_name].y = new_y

    def _norm(self, val, range_min, range_max):
        output = (val - range_min)/(range_max - range_min)
        # if range_max == range_min, we get nan, which we map to 0.5
        if isinstance(output, np.ndarray):
            output[np.isnan(output)] = 0.5
        elif range_min == range_max:
            output = 0.5
        return output

    def _positions(self):
        node_pos = np.array([[node.x,node.y] for node in self.nodes.values()])
        conn_origin = np.array([self.nodes[conn.origin.name].pos for conn in self.graph.connections])
        conn_dest = np.array([self.nodes[conn.dest.name].pos for conn in self.graph.connections])

        range_min = node_pos.min(axis=0)
        range_max = node_pos.max(axis=0)
        node_pos = self._norm(node_pos, range_min, range_max)
        conn_origin = self._norm(conn_origin, range_min, range_max)
        conn_dest = self._norm(conn_dest, range_min, range_max)

        return node_pos, conn_origin, conn_dest

    def draw(self, axes, interval=25):
        self._draw_first(axes)
        self.ani = matplotlib.animation.FuncAnimation(axes.figure, self._update,
                                                      init_func = lambda :self._draw_first(axes),
                                                      interval=interval, blit=False)

    def _draw_first(self, axes):
        axes.clear()
        self._connection_lines.clear()
        self._node_scatter = None

        node_pos, conn_origin, conn_dest = self._positions()

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

    def _update(self,*args):
        self.relax()

        node_pos, conn_origin, conn_dest = self._positions()

        for line,origin,dest in zip(self._connection_lines,conn_origin,conn_dest):
            uniform = np.arange(0, 1, 0.01)

            origin = origin.reshape((2,1))
            dest = dest.reshape((2,1))
            uniform = uniform.reshape((1,len(uniform)))

            pos = origin + (dest-origin)*uniform

            line[0].set_xdata(pos[0])
            line[0].set_ydata(pos[1])

        self._node_scatter.set_offsets(node_pos)


class LayoutNode:
    def __init__(self):
        self.pos = np.array([0,0],dtype='float64')

    @property
    def x(self):
        return self.pos[0]

    @x.setter
    def x(self, val):
        self.pos[0] = val

    @property
    def y(self):
        return self.pos[1]

    @y.setter
    def y(self, val):
        self.pos[1] = val
