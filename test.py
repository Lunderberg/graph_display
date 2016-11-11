#!/usr/bin/env python3

import random

import matplotlib.pyplot as plt

from graph import Graph
from layout import Layout

graph = Graph()

# for i in range(10):
#     graph.add_node(i)

# for i in range(20):
#     graph.add_connection(random.randint(0,9),
#                          random.randint(0,9))

graph.add_node('Input')
graph.add_node('Bias')
graph.add_node('Hidden')
graph.add_node('Output')

graph.add_connection('Input', 'Output')
graph.add_connection('Input', 'Hidden')
graph.add_connection('Bias', 'Output')
graph.add_connection('Bias', 'Hidden')
graph.add_connection('Hidden', 'Output')


layout = Layout(graph)
# layout.add_condition(('fixed_x', 'Input', 0.3))
# layout.add_condition(('fixed_x', 'Bias', 0.6))
layout.add_condition(('fixed_y', 'Input', 0.0))
layout.add_condition(('fixed_y', 'Bias', 0.0))
layout.add_condition(('fixed_y', 'Output', 1.0))
#layout.relax(n_iter = 100, repulsion_constant=0.0)

fig, axes = plt.subplots()
layout.draw(axes)
#plt.show()

plt.show(block=False)
while plt.fignum_exists(fig.number):
    layout.relax()
    layout._update()
    #layout.draw(axes)
    fig.canvas.draw()
#import IPython; IPython.embed()
