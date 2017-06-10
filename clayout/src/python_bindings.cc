#include <sstream>
#include <tuple>

#include "GVector.hh"
#include "Layout.hh"

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/numpy.h"

namespace py = pybind11;

using namespace clayout;

PYBIND11_MODULE(clayout, m) {
  m.doc() = "Layout implemented in C++";

  py::class_<GVector<2> >(m, "GVector2")
    .def(py::init<>())
    .def(py::init<double,double>())
    .def("X", (double& (GVector<2>::*)())&GVector<2>::X)
    .def("Y", (double& (GVector<2>::*)())&GVector<2>::Y);

  py::class_<Layout>(m, "Layout")
    .def(py::init<>())
    .def("add_node", &Layout::add_node)
    .def("add_connection", &Layout::add_connection)
    .def("relax", &Layout::relax)

    .def("fix_x", &Layout::fix_x)
    .def("fix_y", &Layout::fix_y)
    .def("same_x", &Layout::same_x)
    .def("same_y", &Layout::same_y)

    .def("reset_node", &Layout::reset_node)
    .def("reset_edges", &Layout::reset_edges)

    .def("positions",
         [](Layout& inst) -> std::tuple<py::array_t<double>, py::array_t<double> > {
           auto pos = inst.positions();

           auto node_buf = py::buffer_info(
             pos.node_pos.data(),
             sizeof(double),
             py::format_descriptor<double>::format(),
             2,
             { (int)pos.node_pos.size(), 2 },
             { 2*sizeof(double), sizeof(double) }
           );

           auto num_xy_points = pos.connection_points.size();
           auto conn_buf = py::buffer_info(
             pos.connection_points.data(),
             sizeof(double),
             py::format_descriptor<double>::format(),
             3,
             { int(num_xy_points / pos.num_points_per_connection), int(pos.num_points_per_connection), 2 },
             { 2*pos.num_points_per_connection*sizeof(double),
                 2*sizeof(double), sizeof(double) }
           );

           return std::make_tuple(py::array(node_buf),
                                  py::array(conn_buf));
         })

    .def_property("rel_node_size", &Layout::get_rel_node_size, &Layout::set_rel_node_size);
}
