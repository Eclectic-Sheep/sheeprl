#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "node/node.h"
#include "torch/extension.h"
#include "torch/torch.h"

namespace py = pybind11;

PYBIND11_MODULE(_node, m) {
	py::class_<Node::Node>(m, "Node")
		.def(py::init<const float &>())
		.def("set_hidden_state", &Node::Node::set_hidden_state)
		.def("get_hidden_state", &Node::Node::get_hidden_state)
		.def_readwrite("hidden_state", &Node::Node::hidden_state)
		.def_readwrite("prior", &Node::Node::prior)
		.def_readwrite("children", &Node::Node::children);
}
