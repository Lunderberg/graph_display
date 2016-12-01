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

  normalize(output.node_pos, output.connection_points);

  return output;
}

void Layout::adjust_endpoints(std::vector<GVector<2> >& conn_pos, double width, double height) const {
  for(unsigned int i=0; i<conn_pos.size(); i+=(num_control_points+2)) {
    conn_pos[i] = ellipse_intersection(conn_pos[i], conn_pos[i+1], width, height);
    conn_pos[i+num_control_points+1] = ellipse_intersection(conn_pos[i+num_control_points+1],
                                                            conn_pos[i+num_control_points],
                                                            width, height);
  }
}

GVector<2> Layout::ellipse_intersection(GVector<2> p0, GVector<2> p1, double width, double height) const {
  double q = std::pow((p1.Y() - p0.Y())/(p1.X() - p0.X()), 2);
  double xdiff = std::sqrt(width*width*height*height / (4*height*height + 4*width*width*q));
  double ydiff = std::sqrt(width*width*height*height / (4*width*width + 4*height*height/q));

  xdiff = std::copysign(xdiff, p1.X() - p0.X());
  ydiff = std::copysign(ydiff, p1.Y() - p0.Y());

  return {p0.X() + xdiff, p0.Y() + ydiff};
}

void Layout::normalize(std::vector<GVector<2> >& node_pos, std::vector<GVector<2> >& conn_pos) const {
  double xmin = std::numeric_limits<double>::max();
  double xmax = -std::numeric_limits<double>::max();
  double ymin = std::numeric_limits<double>::max();
  double ymax = -std::numeric_limits<double>::max();

  for(auto& pos : node_pos) {
    xmin = std::min(xmin, pos.X());
    xmax = std::max(xmax, pos.X());
    ymin = std::min(ymin, pos.Y());
    ymax = std::max(ymax, pos.Y());
  }

  for(auto& pos : node_pos) {
    pos.X() = (pos.X() - xmin)/(xmax-xmin);
    pos.Y() = (pos.Y() - ymin)/(ymax-ymin);
  }

  adjust_endpoints(conn_pos, rel_node_size*(xmax-xmin), rel_node_size*(ymax-ymin));

  for(auto& pos : conn_pos) {
    pos.X() = (pos.X() - xmin)/(xmax-xmin);
    pos.Y() = (pos.Y() - ymin)/(ymax-ymin);
  }
}
