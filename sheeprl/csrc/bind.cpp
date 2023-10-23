#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "bar/bar.h"
#include "foo/foo.h"
#include "torch/extension.h"
#include "torch/torch.h"

namespace py = pybind11;

PYBIND11_MODULE(_foo, m) {
	py::class_<foo::Foo>(m, "Foo")
		.def(py::init<torch::Tensor>())
		.def("set_tensor", &foo::Foo::set_tensor)
		.def("get_tensor", &foo::Foo::get_tensor)
		.def_readwrite("tensor", &foo::Foo::tensor);
}

PYBIND11_MODULE(_bar, m) {
	py::class_<bar::Bar>(m, "Bar")
		.def(py::init<torch::Tensor>())
		.def("set_tensor", &bar::Bar::set_tensor)
		.def("get_tensor", &bar::Bar::get_tensor)
		.def_readwrite("tensor", &bar::Bar::tensor);
}