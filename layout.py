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
        self.pseudo_gravity_constant = 0.05

        self.num_control_points = 2
        # List of lists of nodes.
        # Each list corresponds to the virtual nodes within one connection.
        self.virtual_nodes = []

    def add_node(self):
        layout_node = LayoutNode()
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
        mass = 0.1/self.num_control_points
        charge = 0.1/self.num_control_points
        return [LayoutNode(initial + num*(final-initial), mass=mass, charge=charge)
                for num in uniform]

    def _all_nodes(self, with_virtual=False):
        yield from self.nodes
        if with_virtual:
            yield from self._all_virtual_nodes()

    def _all_virtual_nodes(self):
        yield from itertools.chain(*self.virtual_nodes)

    def _all_node_pairs(self):
        yield from itertools.combinations(itertools.chain(
            self.nodes, self._all_virtual_nodes()), 2)

    def _connected_pairs(self):
        for (from_node, to_node) in self.connections:
            yield (self.nodes[from_node], self.nodes[to_node])

        for control_points in self.virtual_nodes:
            yield from zip(control_points[1:], control_points[:-1])

        for conn, control_points in zip(self.connections, self.virtual_nodes):
            if control_points:
                yield (self.nodes[conn[0]], control_points[0])
                yield (self.nodes[conn[1]], control_points[-1])

    def relax(self):
        # Electrostatic repulsion between pairs
        for (node_a, node_b) in self._all_node_pairs():
            disp = node_b.pos - node_a.pos
            dist2 = np.dot(disp, disp)
            if dist2 > 0:
                unit_vec = disp/np.sqrt(dist2)
                force = self.repulsion_constant * node_a.charge * node_b.charge * unit_vec / dist2
            else:
                force = np.array([1,0])

            node_a.pos -= force/node_a.mass
            node_b.pos += force/node_b.mass

        # Spring attraction between pairs
        for (node_a, node_b) in self._connected_pairs():
            disp = node_b.pos - node_a.pos
            force = -self.spring_constant * disp

            node_a.pos -= force/node_a.mass
            node_b.pos += force/node_b.mass

        # Pseudo-gravity, constant force toward zero
        for node in self._all_nodes(with_virtual=True):
            disp = node.pos
            node.pos -= self.pseudo_gravity_constant*len(self.nodes)/(1 + np.exp(-np.abs(disp)))

        # Apply conditions
        self._apply_conditions()

    def fix_x(self, node_index, rel_x):
        self.conditions.append(('fixed_x',node_index,rel_x))

    def fix_y(self, node_index, rel_y):
        self.conditions.append(('fixed_y',node_index,rel_y))

    def same_x(self, node_index_a, node_index_b):
        self.conditions.append(('same_x',node_index_a, node_index_b))

    def same_y(self, node_index_a, node_index_b):
        self.conditions.append(('same_y',node_index_a, node_index_b))

    def _apply_conditions(self):
        xmin = min(node.x for node in self._all_nodes())
        xmax = max(node.x for node in self._all_nodes())
        range_x = xmax - xmin

        ymin = min(node.y for node in self._all_nodes())
        ymax = max(node.y for node in self._all_nodes())
        range_y = ymax - ymin

        for condition in self.conditions:
            if condition[0] == 'fixed_x':
                self.nodes[condition[1]].x = xmin + range_x*condition[2]

            elif condition[0] == 'fixed_y':
                self.nodes[condition[1]].y = ymin + range_y*condition[2]

            elif condition[0] == 'same_x':
                new_x = sum(self.nodes[node_name].x for node_name in condition[1:])/len(condition[1:])
                for node_name in condition[1:]:
                    self.nodes[node_name].x = new_x

            elif condition[0] == 'same_y':
                new_y = sum(self.nodes[node_name].y for node_name in condition[1:])/len(condition[1:])
                for node_name in condition[1:]:
                    self.nodes[node_name].y = new_y

    def positions(self):
        """
        Returns the positions of all nodes and connections, ready to draw.
        There are three return values.
        1. node_pos, an array of positions of each node.
           nodes_pos[i][0] is the x position of the i-ith node.
           nodes_pos[i][1] is the y position of the i-ith node.

        2. connections, an array of the positions of the edges.
           connections[i][j][0] is the x position of the j-th point in the i-th connection
           connections[i][j][1] is the y position of the j-th point in the i-th connection
        """
        node_pos = np.array([node.pos for node in self.nodes])
        conn_origin = np.array([self.nodes[from_index].pos for (from_index, to_index) in self.connections])
        conn_dest = np.array([self.nodes[to_index].pos for (from_index, to_index) in self.connections])

        connections = []
        for i,_ in enumerate(self.connections):
            new_spline = self._control_points(i)
            connections.append(new_spline)
        connections = np.array(connections)

        return node_pos, connections

    def _control_points(self, i):
        """
        node_x_size, node_y_size are the size in real units of each node.

        Returns the spline representing the i-th edge.
        retval[j][0] is the x point of the j-th spline point.
        retval[j][1] is the y point of the j-th spline point.
        """
        from_index,to_index = self.connections[i]
        control_points = self.virtual_nodes[i]

        x = [p.x for p in control_points]
        x.insert(0, self.nodes[from_index].x)
        x.append(self.nodes[to_index].x)

        y = [p.y for p in control_points]
        y.insert(0, self.nodes[from_index].y)
        y.append(self.nodes[to_index].y)

        return np.array([x,y]).T



class LayoutNode:
    def __init__(self, pos=None, mass=1.0, charge=1.0):
        if pos is None:
            self.pos = np.array([0,0],dtype='float64')
        else:
            self.pos = np.array(pos, dtype='float64')

        self.mass = mass
        self.charge = charge

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
