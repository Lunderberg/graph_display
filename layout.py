import itertools
import random

import numpy as np

class Layout:
    def __init__(self):
        self.nodes = []
        self.connections = []

        self.conditions = []
        self.spring_constant = 0.01
        self.repulsion_constant = 0.01
        self.pseudo_gravity = 0.05

        self.num_control_points = 0
        # List of lists of nodes.
        # Each list corresponds to the virtual nodes within one connection.
        self.virtual_nodes = []

    def add_node(self):
        layout_node = LayoutPos()
        layout_node.x = random.uniform(0,1)
        layout_node.y = random.uniform(0,1)
        self.nodes.append(layout_node)

    def reset_nodes(self):
        for node in self.nodes:
            node.x = random.uniform(0,1)
            node.y = random.uniform(0,1)

        self.reset_edges()

    def add_connection(self, from_index, to_index):
        self.connections.append( (from_index,to_index) )
        control_points = self._gen_control_points(from_index, to_index)
        self.virtual_nodes.append(control_points)

    def reset_edges(self, num_control_points = None):
        if num_control_points is None:
            num_control_points = self.num_control_points

        self.virtual_nodes = [self._gen_control_points(from_index, to_index)
                              for (from_index, to_index) in self.connections]

    def _gen_control_points(self, from_index, to_index):
        initial = self.nodes[from_index].pos
        final = self.nodes[to_index].pos
        uniform = np.linspace(0, 1, self.num_control_points+2)[1:-1]
        return [LayoutPos(initial + num*(final-initial)) for num in uniform]

    def _all_nodes(self, with_actual=True, with_virtual=False):
        yield from self.nodes
        if with_virtual:
            yield from self._all_virtual_nodes()

    def _all_virtual_nodes(self):
        yield from itertools.chain(*self.virtual_nodes)

    def _all_equal_pairs(self):
        yield from itertools.combinations(self.nodes,2)
        yield from itertools.combinations(self._all_virtual_nodes(), 2)

    def _all_oneway_pairs(self):
        yield from itertools.product(self.nodes, self._all_virtual_nodes())

    def _connection_pairs(self):
        for (from_node, to_node) in self.connections:
            yield (self.nodes[from_node], self.nodes[to_node])

        for control_points in self.virtual_nodes:
            yield from zip(control_points[1:], control_points[:-1])

    def _connections_oneway(self):
        for conn, control_points in zip(self.connections, self.virtual_nodes):
            if control_points:
                yield (self.nodes[conn[0]], control_points[0])
                yield (self.nodes[conn[1]], control_points[-1])

    def add_condition(self, condition):
        self.conditions.append(condition)

    def relax(self, conditions=None):
        conditions = conditions if conditions is not None else []
        all_conditions = itertools.chain(self.conditions, conditions)

        # Electrostatic repulsion between pairs
        for (node_a, node_b) in self._all_equal_pairs():
            disp = node_b.pos - node_a.pos
            dist2 = np.dot(disp, disp)
            if dist2 > 0:
                unit_vec = disp/np.sqrt(dist2)
                force = self.repulsion_constant * unit_vec / dist2
            else:
                force = np.array([1,0])

            node_a.pos -= force
            node_b.pos += force

        # Electrostatic repulsion, one way.
        # Nodes push virtual nodes, but not the other way around.#
        for (node_from, node_to) in self._all_oneway_pairs():
            disp = node_to.pos - node_from.pos
            dist2 = np.dot(disp, disp)
            unit_vec = disp/np.sqrt(dist2)
            force = self.repulsion_constant * unit_vec / dist2
            node_to.pos += force

        # Spring attraction between pairs
        for (node_a, node_b) in self._connection_pairs():
            disp = node_b.pos - node_a.pos
            force = -self.spring_constant * disp

            node_a.pos -= force
            node_b.pos += force

        # Spring attraction, one way.
        # Nodes pull virtual nodes, but not the other way around.
        for (node_from, node_to) in self._connections_oneway():
            disp = node_to.pos - node_from.pos
            force = -self.spring_constant * disp

            node_to.pos -= force

        # Pseudo-gravity, constant force toward zero
        for node in self._all_nodes(with_virtual=True):
            disp = node.pos
            node.pos -= self.pseudo_gravity/(1 + np.exp(-disp))

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

    def positions(self):
        node_pos = np.array([node.pos for node in self.nodes])
        conn_origin = np.array([self.nodes[from_index].pos for (from_index, to_index) in self.connections])
        conn_dest = np.array([self.nodes[to_index].pos for (from_index, to_index) in self.connections])

        connections_x = np.array([(self.nodes[from_index].x,
                                   *[p.x for p in control_points],
                                   self.nodes[to_index].x)
                                  for (from_index,to_index),control_points in zip(self.connections, self.virtual_nodes)])

        connections_y = np.array([(self.nodes[from_index].y,
                                   *[p.y for p in control_points],
                                   self.nodes[to_index].y)
                                  for (from_index,to_index),control_points in zip(self.connections, self.virtual_nodes)])

        range_min = node_pos.min(axis=0)
        range_max = node_pos.max(axis=0)

        xmin,ymin = range_min
        xmax,ymax = range_max

        node_pos = self._norm(node_pos, range_min, range_max)
        connections_x = self._norm(connections_x, xmin, xmax)
        connections_y = self._norm(connections_y, ymin, ymax)

        return node_pos, connections_x, connections_y


class LayoutPos:
    def __init__(self, pos=None):
        if pos is None:
            self.pos = np.array([0,0],dtype='float64')
        else:
            self.pos = np.array(pos, dtype='float64')

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
