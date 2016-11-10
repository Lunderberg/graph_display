#!/usr/bin/env python3

import random
from itertools import combinations
from enum import Enum

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# Currently from here. http://stackoverflow.com/a/27666700/2689797
# Replace with this one? http://stackoverflow.com/a/39782685/2689797
def add_arrow_to_line2D(axes, line, arrow_locs=[0,2, 0.4, 0.6, 0.8],
                        arrowstyle='-|>', arrowsize=1, transform=None):
    if not isinstance(line, list) or not (isinstance(line[0], mlines.Line2D)):
        raise ValueError('Expected a matpliblib.lines.Line2D object')

    x = line[0].get_xdata()
    y = line[0].get_ydata()

    arrow_kw = dict(arrowstyle=arrowstyle, mutation_scale=10*arrowsize)

    color = line[0].get_color()
    use_multicolor_lines = isinstance(color, np.ndarray)

    if use_multicolor_lines:
        raise NotImplementedError('multicolor lines not supported')
    else:
        arrow_kw['color'] = color

    linewidth = line[0].get_linewidth()
    if isinstance(linewidth, np.ndarray):
        raise NotImplementedError('multiwidth lines not supported')
    else:
        arrow_kw['linewidth'] = linewidth

    if transform is None:
        transform = axes.transData

    arrows = []
    cumsum = np.cumsum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
    for loc in arrow_locs:
        n = np.searchsorted(cumsum, cumsum[-1]*loc)
        arrow_tail = (x[n], y[n])
        arrow_head = (np.mean(x[n:n+2]), np.mean(y[n:n+2]))
        patch = mpatches.FancyArrowPatch(arrow_tail, arrow_head, transform=transform,
                                         **arrow_kw)
        axes.add_patch(patch)
        arrows.append(patch)
    return arrows

class NodeType(Enum):
    input = 0
    bias = 1
    hidden = 2
    output = 3


class Node:
    def __init__(self, name, node_type = NodeType.hidden):
        self.name = name
        self.pos = np.array([0,0],dtype='float64')
        self.node_type = node_type

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

class Graph:
    def __init__(self):
        self.nodes = {}
        self.connections = []

    def add_node(self, node):
        if isinstance(node, Node):
            self.nodes[node.name] = node
        else:
            self.nodes[node] = Node(node)

    def add_connection(self, from_name, to_name):
        self.connections.append((from_name, to_name))

    def randomize_layout(self):
        num_inputs = sum(1 for node in self.nodes.values()
                         if node.node_type==NodeType.input)
        num_outputs = sum(1 for node in self.nodes.values()
                          if node.node_type==NodeType.output)

        i_input = 0
        i_output = 0
        for node in self.nodes.values():
            if node.node_type == NodeType.input:
                node.x = (i_input + 1)/(num_inputs + 1)
                node.y = 0
                i_input += 1
            elif node.node_type == NodeType.output:
                node.x = (i_output + 1)/(num_outputs + 1)
                node.y = 1
                i_output += 1
            else:
                node.x = random.uniform(0,1)
                node.y = random.uniform(0,1)

    def spring_layout(self, iterations=500, spring_constant=0.01, repulsion_constant=0.01):
        for i in range(iterations):
            self._spring_layout_step(spring_constant, repulsion_constant)

    def _spring_layout_step(self, spring_constant, repulsion_constant):
        forces = {name:np.array([0,0], dtype='float64') for name in self.nodes}

        # Electrostatic repulsion
        for (name_a, name_b) in combinations(self.nodes, 2):
            node_a = self.nodes[name_a]
            node_b = self.nodes[name_b]
            disp = node_a.pos - node_b.pos
            dist2 = np.dot(disp, disp)
            unit_vec = disp/np.sqrt(dist2)
            force = repulsion_constant * unit_vec / dist2
            forces[name_a] += force
            forces[name_b] -= force

        # Spring attraction
        for (from_name, to_name) in self.connections:
            disp = self.nodes[to_name].pos - self.nodes[from_name].pos
            force = -spring_constant * disp

            forces[to_name] += force
            forces[from_name] -= force

        # Input/output nodes have fixed x values, identical y values
        force_y_input = []
        force_y_output = []
        for node in self.nodes.values():
            if node.node_type==NodeType.hidden:
                node.pos += forces[node.name]
            elif node.node_type==NodeType.input:
                force_y_input.append(forces[node.name][0])
            elif node.node_type==NodeType.output:
                force_y_output.append(forces[node.name][0])

        force_y_input = sum(force_y_input)/len(force_y_input)
        force_y_output = sum(force_y_output)/len(force_y_output)
        for node in self.nodes.values():
            if node.node_type == NodeType.input:
                node.y += force_y_input
            elif node.node_type == NodeType.output:
                node.y += force_y_output

    def draw(self, axes):
        for (from_name, to_name) in self.connections:
            x0 = self.nodes[from_name].x
            y0 = self.nodes[from_name].y
            xf = self.nodes[to_name].x
            yf = self.nodes[to_name].y

            uniform = np.arange(0, 1, 0.01)
            line = axes.plot(x0 + (xf-x0)*uniform,
                             y0 + (yf-y0)*uniform,
                             zorder=1)
            add_arrow_to_line2D(axes, line, arrow_locs=np.linspace(0, 1, 25))

        color_map = {NodeType.input:'green',
                     NodeType.hidden:'blue',
                     NodeType.output:'red'}
        for node_type in [NodeType.input,NodeType.hidden,NodeType.output]:
            xvals = [node.x for node in self.nodes.values() if node.node_type==node_type]
            yvals = [node.y for node in self.nodes.values() if node.node_type==node_type]
            sizes = [1000 for node in self.nodes.values() if node.node_type==node_type]
            axes.scatter(xvals, yvals, s=sizes, zorder=2, color=color_map[node_type], edgecolor='black')


graph = Graph()

for i in range(10):
    node_type = (NodeType.input if i < 2 else
                 NodeType.output if i<4 else
                 NodeType.hidden)
    graph.add_node(Node(i,node_type))

for i in range(20):
    graph.add_connection(random.randint(0,9),
                         random.randint(0,9))
# graph.add_node(Node('Input',NodeType.input))
# graph.add_node(Node('Bias',NodeType.input))
# graph.add_node(Node('Hidden',NodeType.hidden))
# graph.add_node(Node('Output',NodeType.output))

# graph.add_connection('Input', 'Output')
# graph.add_connection('Input', 'Hidden')
# graph.add_connection('Bias', 'Output')
# graph.add_connection('Bias', 'Hidden')
# graph.add_connection('Hidden', 'Output')

graph.randomize_layout()
graph.spring_layout()

fig, axes = plt.subplots()
graph.draw(axes)
plt.show()

#plt.show(block=False)
#import IPython; IPython.embed()
