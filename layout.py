import itertools
import random

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


class Layout:
    def __init__(self, graph):
        self.graph = graph
        self.nodes = {}
        self.conditions = []
        self.spring_constant = 0.01
        self.repulsion_constant = 0.01

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

    def relax(self, n_iter=1, conditions=None, spring_constant=None, repulsion_constant=None):
        for _ in range(n_iter):
            if spring_constant is None:
                spring_constant = self.spring_constant

            if repulsion_constant is None:
                repulsion_constant = self.repulsion_constant

            all_conditions = all_conditions = itertools.chain(
                self.conditions, conditions if conditions is not None else [])


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

    def _norm(val, range_min, range_max):
        if range_min == range_max:
            # Returns 0.5 for float, array of 0.5 for array
            return 0.5 * val/val
        else:
            return (val - range_min)/(range_max - range_min)

    def draw(self, axes):
        x = [node.x for node in self.nodes.values()]
        y = [node.y for node in self.nodes.values()]

        x_range = min(x), max(x)
        y_range = min(y), max(y)

        x = self._norm(x, *x_range)
        y = self._norm(y, *y_range)

        axes.clear()
        for conn in self.graph.connections:
            from_name = conn.origin.name
            to_name = conn.dest.name
            x0 = self.nodes[from_name].x
            y0 = self.nodes[from_name].y
            xf = self.nodes[to_name].x
            yf = self.nodes[to_name].y

            x0 = self._norm(x0, *x_range)
            xf = self._norm(xf, *x_range)
            y0 = self._norm(y0, *x_range)
            yf = self._norm(yf, *x_range)

            uniform = np.arange(0, 1, 0.01)
            line = axes.plot(x0 + (xf-x0)*uniform,
                             y0 + (yf-y0)*uniform,
                             zorder=1)
            add_arrow_to_line2D(axes, line, arrow_locs=np.linspace(0, 1, 25))


        axes.scatter(x, y,
                     sizes = [1000 for node in self.nodes.values()],
                     zorder = 2,
                     edgecolor = 'black')

        axes.set_xlim(-1.1, 1.1)
        axes.set_ylim(-1.1, 1.1)

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
