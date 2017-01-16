#ifndef _LAYOUT_H_
#define _LAYOUT_H_

#include <random>
#include <utility>
#include <vector>

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

  // Modifiers
  void add_node();
  void add_connection(int from_index, int to_index);

  // Move toward equilibrium.
  void relax();

  // Constraints
  void fix_x(int index, double x_pos) { fixed_x_pos.push_back({index,x_pos}); }
  void fix_y(int index, double y_pos) { fixed_y_pos.push_back({index,y_pos}); }
  void same_x(int index_a, int index_b) { same_x_pos.push_back({index_a,index_b}); }
  void same_y(int index_a, int index_b) { same_y_pos.push_back({index_a,index_b}); }

  // Reset portions of the graph.
  void reset_node();
  void reset_edges();

  // Relative size of the nodes
  void set_rel_node_size(double val) { rel_node_size = val; }
  double get_rel_node_size() const { return rel_node_size; }

  // Return the locations of everything
  DrawingPositions positions() const;

private:
  void gen_control_points(Connection& conn);
  void electrostatic(Node& a, Node& b);
  void spring(Node& a, Node& b);
  void pseudo_gravity(Node& a);

  void apply_constraints();

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

  std::vector<std::pair<int, double> > fixed_x_pos;
  std::vector<std::pair<int, double> > fixed_y_pos;
  std::vector<std::pair<int, int> > same_x_pos;
  std::vector<std::pair<int, int> > same_y_pos;
};

#endif /* _LAYOUT_H_ */
