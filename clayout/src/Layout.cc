#include "Layout.hh"

#include <algorithm>
#include <iostream>

Layout::Layout()
  : gen(std::random_device()()),
    spring_constant(0.01), repulsion_constant(0.01), pseudo_gravity_constant(0.01),
    num_control_points(2), num_spline_points(10), rel_node_size(0.05) { }

void Layout::add_node() {
  Node new_node;
  new_node.pos.X() = std::uniform_real_distribution<>(0,1)(gen);
  new_node.pos.Y() = std::uniform_real_distribution<>(0,1)(gen);
  new_node.mass = 1.0;
  new_node.charge = 1.0;
  nodes.push_back(new_node);
}

void Layout::add_connection(int from_index, int to_index) {
  Connection new_connection;
  new_connection.from_index = from_index;
  new_connection.to_index = to_index;
  gen_control_points(new_connection);
  connections.push_back(new_connection);
}

void Layout::gen_control_points(Connection& conn) {
  const GVector<2> initial = nodes[conn.from_index].pos;
  const GVector<2> final = nodes[conn.to_index].pos;
  conn.virtual_node_index = virtual_nodes.size();

  for(int i=0; i<num_control_points; i++) {
    Node control;
    control.pos = initial + (final-initial)*((i+1.0)/(num_control_points+1.0));
    control.mass = 0.1/num_control_points;
    control.charge = 0.1/num_control_points;
    virtual_nodes.push_back(control);
  }
}

void Layout::reset_node() {
  for(auto& node : nodes) {
    node.pos.X() = std::uniform_real_distribution<>(0,1)(gen);
    node.pos.Y() = std::uniform_real_distribution<>(0,1)(gen);
  }
  reset_edges();
}

void Layout::reset_edges() {
  virtual_nodes.clear();
  for(auto& conn : connections) {
    gen_control_points(conn);
  }
}

void Layout::relax() {
  for(unsigned int i=0; i<nodes.size(); i++) {
    for(unsigned int j=1; j<nodes.size(); j++) {
      electrostatic(nodes[i], nodes[j]);
    }
  }

  for(unsigned int i=0; i<virtual_nodes.size(); i++) {
    for(unsigned int j=1; j<virtual_nodes.size(); j++) {
      electrostatic(virtual_nodes[i], virtual_nodes[j]);
    }
  }

  for(auto& node : nodes) {
    for(auto& vnode : virtual_nodes) {
      electrostatic(node, vnode);
    }
  }


  for(auto& conn : connections) {
    spring(nodes[conn.from_index], nodes[conn.to_index]);
    if(num_control_points) {
      spring(nodes[conn.from_index], virtual_nodes[conn.virtual_node_index]);
      spring(nodes[conn.to_index], virtual_nodes[conn.virtual_node_index + num_control_points-1]);
    }
    for(int i=0; i+1<num_control_points; i++) {
      spring(virtual_nodes[conn.virtual_node_index],
             virtual_nodes[conn.virtual_node_index+1]);
    }
  }

  for(auto& node : nodes) {
    pseudo_gravity(node);
  }

  apply_constraints();
}

void Layout::electrostatic(Node& a, Node& b) {
  auto disp = b.pos - a.pos;
  auto dist2 = disp*disp;

  GVector<2> force;
  if(dist2 > 0) {
    force = repulsion_constant * a.charge * b.charge * disp.UnitVector() / dist2;
  } else {
    force = {1,0};
  }
  a.pos -= force/a.mass;
  b.pos += force/b.mass;
}

void Layout::spring(Node& a, Node& b) {
  auto disp = b.pos - a.pos;
  auto force = -spring_constant * disp;

  a.pos -= force/a.mass;
  b.pos += force/b.mass;
}

void Layout::pseudo_gravity(Node& a) {
  auto disp = a.pos;
  // auto force = pseudo_gravity_constant * nodes.size() *
  //   disp.UnitVector() / (1 + std::exp(-disp.Mag()));

  auto force = -pseudo_gravity_constant * nodes.size() * disp.UnitVector()
    / (1 + 1/disp.Mag());

  a.pos += force;
}

void Layout::apply_constraints() {
  // Fixed x positions
  auto x_elements = std::minmax_element(nodes.begin(), nodes.end(),
                                        [](const Node& a, const Node& b) { return a.pos.X() < b.pos.X(); } );
  double xmin = x_elements.first->pos.X();
  double xmax = x_elements.second->pos.X();
  double xrange = xmax - xmin;
  for(auto& con : fixed_x_pos) {
    nodes[con.first].pos.X() = xmin + xrange*con.second;
  }

  // Fixed y positions
  auto y_elements = std::minmax_element(nodes.begin(), nodes.end(),
                                        [](const Node& a, const Node& b) { return a.pos.Y() < b.pos.Y(); } );
  double ymin = y_elements.first->pos.Y();
  double ymax = y_elements.second->pos.Y();
  double yrange = ymax - ymin;
  for(auto& con : fixed_y_pos) {
    nodes[con.first].pos.Y() = ymin + yrange*con.second;
  }

  // Same x positions
  for(auto& con : same_x_pos) {
    double new_x = (nodes[con.first].pos.X() + nodes[con.second].pos.X())/2;
    nodes[con.first].pos.X() = new_x;
    nodes[con.second].pos.X() = new_x;
  }

  // Same y positions
  for(auto& con : same_y_pos) {
    double new_y = (nodes[con.first].pos.Y() + nodes[con.second].pos.Y())/2;
    nodes[con.first].pos.Y() = new_y;
    nodes[con.second].pos.Y() = new_y;
  }
}

DrawingPositions Layout::positions() const {
  DrawingPositions output;

  output.node_pos.reserve(nodes.size());
  for(auto& node : nodes) {
    output.node_pos.push_back(node.pos);
  }

  output.num_points_per_connection = num_control_points+2;
  output.connection_points.reserve((num_control_points+2) * connections.size());
  for(auto& conn : connections) {
    output.connection_points.push_back(nodes[conn.from_index].pos);
    for(int i=0; i<num_control_points; i++) {
      output.connection_points.push_back(virtual_nodes[conn.virtual_node_index+i].pos);
    }
    output.connection_points.push_back(nodes[conn.to_index].pos);
  }

  return output;
}
