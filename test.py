#!/usr/bin/env python3

import random

import matplotlib
import matplotlib.pyplot as plt

from graph import Graph

graph = Graph()

# for i in range(10):
#     graph.add_node(i, 'red' if random.random() < 0.2 else 'blue')

# for i in range(20):
#     graph.add_connection(random.randint(0,9),
#                          random.randint(0,9),
#                          boxed=random.random() < 0.2)

graph.add_node('Input', color='green')
graph.add_node('Bias', color='green')
graph.add_node('Hidden')
graph.add_node('Output', color='red')

graph.add_connection('Input', 'Output')
graph.add_connection('Input', 'Hidden')
graph.add_connection('Bias', 'Output')
graph.add_connection('Bias', 'Hidden')
graph.add_connection('Hidden', 'Output')

# graph.same_y('Input','Bias')
# graph.same_x('Input','Output')

graph.fix_y('Input', 0.0)
graph.fix_y('Bias', 0.0)
graph.fix_y('Output', 1.0)

fig, axes = plt.subplots()
fig.set_facecolor('white')
graph.draw(axes)
plt.show()

# plt.show(block=False)
# while plt.fignum_exists(fig.number):
#     layout.relax()
#     layout._update()
#     #layout.draw(axes)
#     fig.canvas.draw()
#import IPython; IPython.embed()
