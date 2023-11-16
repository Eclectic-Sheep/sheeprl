#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "node/mcts.h"
#include "torch/extension.h"
#include "torch/torch.h"

namespace py = pybind11;

PYBIND11_MODULE(_node, m) {
	py::class_<Node::Node>(m, "Node")
		.def(py::init<const float &>())
		.def("set_hidden_state", &Node::Node::set_hidden_state)
		.def("get_hidden_state", &Node::Node::get_hidden_state)
		.def("expand", &Node::Node::expand)
		.def("add_exploration_noise", &Node::Node::add_exploration_noise)
		.def("value", &Node::Node::value)
		.def_readwrite("hidden_state", &Node::Node::hidden_state)
		.def_readwrite("prior", &Node::Node::prior)
		.def_readwrite("reward", &Node::Node::reward)
		.def_readwrite("visit_count", &Node::Node::visit_count)
		.def_readwrite("children", &Node::Node::children)
		.def_readwrite("imagined_action", &Node::Node::imagined_action);

    py::class_<mcts::MinMaxStats>(m, "MinMaxStats")
        .def(py::init<>())
        .def("update", &mcts::MinMaxStats::update)
        .def("normalize", &mcts::MinMaxStats::normalize);

    m.def("rollout", &mcts::rollout, "Rollout function", py::return_value_policy::reference);
    m.def("backpropagate", &mcts::backpropagate, "Backpropagation function");
    m.def("forward_model", &tester::forward_model, "Forward model function");
}