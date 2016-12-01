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

        # Size relative to the full extent between min and max in x,y
        self.rel_node_size = 0.05

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

    def _all_nodes(self, with_actual=True, with_virtual=False):
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

    def add_condition(self, condition):
        self.conditions.append(condition)

    def relax(self, conditions=None):
        conditions = conditions if conditions is not None else []
        all_conditions = itertools.chain(self.conditions, conditions)

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
        center = (range_min + range_max)/2.0
        diff = (range_max - range_min)
        range_min = center - 0.55*diff
        range_max = center + 0.55*diff

        output = (val - range_min)/(range_max - range_min)
        # if range_max == range_min, we get nan, which we map to 0.5
        if isinstance(output, np.ndarray):
            output[np.isnan(output)] = 0.5
        elif range_min == range_max:
            output = 0.5
        return output

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

        xmin = node_pos[:,0].min()
        xmax = node_pos[:,0].max()
        ymin = node_pos[:,1].min()
        ymax = node_pos[:,1].max()

        range_min = np.array([xmin,ymin])
        range_max = np.array([xmax,ymax])

        connections = []
        for i,_ in enumerate(self.connections):
            new_spline = self._spline(i,
                                      self.rel_node_size*(xmax-xmin),
                                      self.rel_node_size*(ymax-ymin))
            connections.append(new_spline)

        connections = np.array(connections)


        node_pos = self._norm(node_pos, range_min, range_max)
        connections = self._norm(connections, range_min, range_max)

        return node_pos, connections

    def _spline(self, i, node_x_size, node_y_size):
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
        x = np.array(x)

        y = [p.y for p in control_points]
        y.insert(0, self.nodes[from_index].y)
        y.append(self.nodes[to_index].y)
        y = np.array(y)

        p0 = self._ellipse_intersection( (x[0],y[0]), (x[1],y[1]),
                                         node_x_size, node_y_size)
        pf = self._ellipse_intersection( (x[-1],y[-1]), (x[-2],y[-2]),
                                         node_x_size, node_y_size)
        x[0] = p0[0]
        y[0] = p0[1]
        x[-1] = pf[0]
        y[-1] = pf[1]



        return np.array([x,y]).T

    def _ellipse_intersection(self, ellipse_center, outside_point, width, height):
        x0,y0 = ellipse_center
        x1,y1 = outside_point

        if x0==x1 and y0==y1:
            return ellipse_center

        q = ((y1-y0)/(x1-x0))**2
        xdiff = np.sqrt(width**2*height**2/(4*height**2 + 4*width**2*q))
        ydiff = np.sqrt(width**2*height**2/(4*width**2 + 4*height**2/q))

        xdiff *= np.sign(x1-x0)
        ydiff *= np.sign(y1-y0)

        return (x0+xdiff, y0+ydiff)





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
