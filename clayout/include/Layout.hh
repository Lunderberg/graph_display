#ifndef _LAYOUT_H_
#define _LAYOUT_H_

#include <vector>
#include <random>

#include "GVector.hh"

struct Node {
  Node()
    : pos(), mass(1), charge(1) { }

  GVector<2> pos;
  double mass;
  double charge;
};

struct Connection {
  Connection()
    : from_index(-1), to_index(-1), virtual_node_index(-1) { }

  int from_index;
  int to_index;
  int virtual_node_index;
};

struct DrawingPositions {
  std::vector<GVector<2> > node_pos;
  std::vector<GVector<2> > connection_points;
  unsigned int num_points_per_connection;
};

class Layout {
public:
  Layout();

  void add_node();
  void add_connection(int from_index, int to_index);
  void relax();

  void reset_node();
  void reset_edges();

  void set_rel_node_size(double val) { rel_node_size = val; }
  double get_rel_node_size() const { return rel_node_size; }

  DrawingPositions positions() const;

private:
  void gen_control_points(Connection& conn);
  void electrostatic(Node& a, Node& b);
  void spring(Node& a, Node& b);
  void pseudo_gravity(Node& a);

  std::vector<GVector<2> > spline(const Connection& conn) const;
  void normalize(std::vector<GVector<2> >& node_pos, std::vector<GVector<2> >& conn_pos) const;

  std::mt19937 gen;

  double spring_constant;
  double repulsion_constant;
  double pseudo_gravity_constant;

  int num_control_points;
  int num_spline_points;

  double rel_node_size;

  std::vector<Node> nodes;
  std::vector<Node> virtual_nodes;
  std::vector<Connection> connections;
};

#endif /* _LAYOUT_H_ */
