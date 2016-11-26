#!/usr/bin/env python3

import clayout

t = clayout.Layout()

t.add_node()
t.add_node()

t.add_connection(0,1)

print(t.positions())
