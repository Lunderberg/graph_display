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

    def add_node(self):
        layout_node = LayoutNode()
        layout_node.x = random.uniform(0,1)
        layout_node.y = random.uniform(0,1)
        self.nodes.append(layout_node)

    def add_connection(self, from_index, to_index):
        self.connections.append( (from_index,to_index) )

    def reset_nodes(self):
        for node in self.nodes:
            node.x = random.uniform(0,1)
            node.y = random.uniform(0,1)

    def reset_edges(self):
        raise NotImplementedError('reset_edges not implemented')

    def add_condition(self, condition):
        self.conditions.append(condition)

    def relax(self, n_iter=1, conditions=None):
        conditions = conditions if conditions is not None else []
        all_conditions = all_conditions = itertools.chain(self.conditions, conditions)

        for _ in range(n_iter):
            forces = [np.array([0,0], dtype='float64') for node in self.nodes]

            # Electrostatic repulsion
            for ((i_a,node_a), (i_b,node_b)) in itertools.combinations(enumerate(self.nodes), 2):
                disp = node_b.pos - node_a.pos
                dist2 = np.dot(disp, disp)
                unit_vec = disp/np.sqrt(dist2)
                force = self.repulsion_constant * unit_vec / dist2
                forces[i_a] -= force
                forces[i_b] += force

            # Spring attraction
            for (from_index, to_index) in self.connections:
                disp = self.nodes[to_index].pos - self.nodes[from_index].pos
                force = -self.spring_constant * disp

                forces[to_index] += force
                forces[from_index] -= force

            # Pseudo-gravity, constant force toward zero
            for i,node in enumerate(self.nodes):
                disp = node.pos
                forces[i] -= self.pseudo_gravity/(1 + np.exp(-disp))

            # Update node positions
            for force,node in zip(forces, self.nodes):
                node.pos += force

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
        conn_origin = np.array([self.nodes[from_index].pos for (from_index,to_index) in self.connections])
        conn_dest = np.array([self.nodes[to_index].pos for (from_index,to_index) in self.connections])

        range_min = node_pos.min(axis=0)
        range_max = node_pos.max(axis=0)
        node_pos = self._norm(node_pos, range_min, range_max)
        conn_origin = self._norm(conn_origin, range_min, range_max)
        conn_dest = self._norm(conn_dest, range_min, range_max)

        return node_pos, conn_origin, conn_dest


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
